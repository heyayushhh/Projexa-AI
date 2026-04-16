from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, BackgroundTasks
from typing import List
from ..utils.models import SpeechAnalysisInDB, TrainingSessionInDB, DashboardStatsResponse, TrendDay
from ..utils.db import get_db
from bson import ObjectId
from datetime import datetime, timedelta, time as dt_time
from ..utils.auth import get_current_user_id
import uuid
import os
import logging

from ..services.ml_service import run_stutter_analysis

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/stats")
async def get_dashboard_stats(user_id: str = Depends(get_current_user_id)):
    db = get_db()
    
    # 1. Total Completed Training Sessions (Unique Levels Mastered)
    # MUST have at least one level correctly done to count
    pipeline_unique = [
        {"$match": {"user_id": user_id, "is_correct": True}},
        {"$group": {"_id": "$exercise_id"}}
    ]
    unique_levels = await db.training_sessions.aggregate(pipeline_unique).to_list(None)
    total_sessions = len(unique_levels)
    
    # 2. Average Fluency Score (from all sessions, speech + training)
    # Including all attempts to be more realistic
    pipeline_score = [
        {"$match": {"user_id": user_id}},
        {"$group": {"_id": None, "avg_score": {"$avg": "$fluency_score"}}}
    ]
    score_data = await db.training_sessions.aggregate(pipeline_score).to_list(None)
    
    # Also get speech analysis average
    speech_score_data = await db.speech_analysis.aggregate(pipeline_score).to_list(None)
    
    scores = []
    if score_data: scores.append(score_data[0]["avg_score"])
    if speech_score_data: scores.append(speech_score_data[0]["avg_score"])
    
    avg_score = round(sum(scores) / len(scores)) if scores else 0
    
    # 3. Streak Calculation
    streak = await calculate_streak(user_id)
    
    return {
        "total_sessions": total_sessions,
        "avg_score": avg_score,
        "streak": streak,
        "fluency_trend": await get_fluency_trend(user_id)
    }

async def get_fluency_trend(user_id: str) -> List[dict]:
    db = get_db()
    # Get last 7 days of average scores
    trend = []
    today = datetime.utcnow().date()
    
    for i in range(6, -1, -1):
        target_date = today - timedelta(days=i)
        start_dt = datetime.combine(target_date, dt_time.min)
        end_dt = datetime.combine(target_date, dt_time.max)
        
        # Aggregate from both collections
        pipeline = [
            {"$match": {
                "user_id": user_id,
                "created_at": {"$gte": start_dt, "$lte": end_dt}
            }},
            {"$group": {"_id": None, "avg": {"$avg": "$fluency_score"}}}
        ]
        
        training_res = await db.training_sessions.aggregate(pipeline).to_list(None)
        speech_res = await db.speech_analysis.aggregate(pipeline).to_list(None)
        
        scores = []
        if training_res: scores.append(training_res[0]["avg"])
        if speech_res: scores.append(speech_res[0]["avg"])
        
        day_avg = round(sum(scores) / len(scores)) if scores else 0
        
        trend.append({
            "day": target_date.strftime("%a"), # e.g. "Mon"
            "score": day_avg
        })
        
    return trend

async def calculate_streak(user_id: str) -> int:
    db = get_db()
    
    # Get all unique dates with activity
    # Combine training sessions and speech analysis
    pipeline = [
        {"$match": {"user_id": user_id}},
        {"$project": {
            "date": {
                "$dateToString": {
                    "format": "%Y-%m-%d",
                    "date": "$created_at"
                }
            }
        }},
        {"$group": {"_id": "$date"}},
        {"$sort": {"_id": -1}}
    ]
    
    training_dates = await db.training_sessions.aggregate(pipeline).to_list(None)
    speech_dates = await db.speech_analysis.aggregate(pipeline).to_list(None)
    
    all_dates = sorted(list(set([d["_id"] for d in training_dates] + [d["_id"] for d in speech_dates])), reverse=True)
    
    if not all_dates:
        return 0
        
    today = datetime.utcnow().strftime("%Y-%m-%d")
    yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    # If the latest activity is not today or yesterday, streak is broken
    if all_dates[0] not in [today, yesterday]:
        return 0
        
    streak = 0
    current_date = datetime.strptime(all_dates[0], "%Y-%m-%d")
    
    for date_str in all_dates:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        expected_date = current_date - timedelta(days=streak)
        
        if date_obj.date() == expected_date.date():
            streak += 1
        else:
            break
            
    return streak

@router.get("/sessions")
async def get_recent_sessions(user_id: str = Depends(get_current_user_id)):
    db = get_db()
    
    # 1. Fetch Speech Analysis sessions
    speech_sessions = await db.speech_analysis.find({"user_id": user_id}).sort("created_at", -1).to_list(10)
    for s in speech_sessions:
        s["_id"] = str(s["_id"])
        s["session_type"] = "speech_analysis"
        if "created_at" not in s:
            s["created_at"] = datetime.utcnow()
        
    # 2. Fetch Training sessions
    training_sessions = await db.training_sessions.find({"user_id": user_id}).sort("created_at", -1).to_list(10)
    for s in training_sessions:
        s["_id"] = str(s["_id"])
        s["session_type"] = "training"
        if "created_at" not in s:
            s["created_at"] = datetime.utcnow()
        # Map fields for UI consistency
        s["fluency_score"] = s.get("fluency_score", 0)
        
    # 3. Merge and sort - with safe fallback for created_at
    all_sessions = sorted(
        speech_sessions + training_sessions, 
        key=lambda x: x.get("created_at", datetime.min), 
        reverse=True
    )[:10]
    return all_sessions

@router.post("/analyze")
async def analyze_speech(background_tasks: BackgroundTasks, file: UploadFile = File(...), user_id: str = Depends(get_current_user_id)):
    session_id = str(uuid.uuid4())
    temp_filename = f"temp_{session_id}_{file.filename}"
    
    with open(temp_filename, "wb") as f:
        content = await file.read()
        f.write(content)
    
    background_tasks.add_task(process_and_store, user_id, temp_filename)
    return {"message": "Analysis started", "session_id": session_id}

@router.post("/training/complete")
async def complete_training(
    exercise_id: int, 
    user_id: str = Depends(get_current_user_id),
    session_data: dict = None
):
    db = get_db()
    
    training_data = {
        "user_id": user_id,
        "exercise_id": exercise_id,
        "completed": True,
        "created_at": datetime.utcnow()
    }
    
    if session_data:
        training_data.update({
            "spoken_text": session_data.get("spokenText"),
            "expected_text": session_data.get("expectedText"),
            "fluency_score": session_data.get("fluencyScore"),
            "is_correct": session_data.get("isCorrect"),
            "word_count": session_data.get("wordCount"),
            "error_count": session_data.get("errorCount"),
            "type": session_data.get("type"),
            "difficulty": session_data.get("difficulty"),
            "level": session_data.get("level")
        })
    
    # Use update_one with upsert=True to overwrite the latest score for this specific exercise
    # This ensures "what ever the new score will come will be the new score for that level"
    await db.training_sessions.update_one(
        {"user_id": user_id, "exercise_id": exercise_id},
        {"$set": training_data},
        upsert=True
    )
    
    # Update user progress
    if session_data and session_data.get("isCorrect"):
        await db.user_progress.update_one(
            {"user_id": user_id, "type": session_data.get("type")},
            {"$set": {
                "last_completed_level": session_data.get("level"),
                "last_completed_difficulty": session_data.get("difficulty"),
                "updated_at": datetime.utcnow()
            }},
            upsert=True
        )
        
    return {"message": "Training session saved"}

@router.get("/training/progress")
async def get_training_progress(user_id: str = Depends(get_current_user_id)):
    db = get_db()
    # Get all training sessions for this user to see scores
    sessions = await db.training_sessions.find(
        {"user_id": user_id},
        {"exercise_id": 1, "fluency_score": 1, "is_correct": 1}
    ).to_list(None)
    
    # Format as a dictionary for easier frontend lookup: { exercise_id: { score, is_correct } }
    # 🔥 Ensure exercise_id is treated as a number for the key
    progress_map = {}
    for s in sessions:
        try:
            ex_id = int(s["exercise_id"])
            progress_map[ex_id] = {
                "score": s.get("fluency_score", 0),
                "is_correct": s.get("is_correct", False)
            }
        except (ValueError, TypeError):
            continue
    
    return {"progress": progress_map}

async def process_and_store(user_id: str, file_path: str):
    db = get_db()
    try:
        # Call the ML service logic
        analysis_result = await run_stutter_analysis(file_path)
        
        result = {
            "user_id": user_id,
            "transcript": analysis_result["transcript"],
            "fluency_score": analysis_result["fluencyScore"],
            "total_words": analysis_result["totalWords"],
            "stutterEvents": analysis_result["stutterEvents"],
            "headMovements": analysis_result["headMovements"],
            "created_at": datetime.utcnow()
        }
        
        await db.speech_analysis.insert_one(result)
        
    except Exception:
        logger.exception("Processing error")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

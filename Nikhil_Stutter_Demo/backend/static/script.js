let mediaRecorder;
let recordedChunks = [];

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const preview = document.getElementById("preview");

startBtn.addEventListener("click", async () => {
    console.log("Recording started");

    const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true
    });

    preview.srcObject = stream;

    mediaRecorder = new MediaRecorder(stream);
    recordedChunks = [];

    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            recordedChunks.push(event.data);
        }
    };

    mediaRecorder.onstop = async () => {
        console.log("Recording stopped");
        await uploadVideo();
    };

    mediaRecorder.start();

    startBtn.disabled = true;
    stopBtn.disabled = false;
});

stopBtn.addEventListener("click", () => {
    console.log("Stop clicked");

    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
    }

    startBtn.disabled = false;
    stopBtn.disabled = true;
});

async function uploadVideo() {
    console.log("Uploading...");

    const blob = new Blob(recordedChunks, { type: "video/webm" });

    const formData = new FormData();
    formData.append("video", blob, "recording.webm");

    const response = await fetch("/upload", {
        method: "POST",
        body: formData
    });

    const data = await response.json();

    console.log("Response:", data);

    localStorage.setItem("last_session", data.session_id);

    window.location.href = `/result?session=${data.session_id}`;
}
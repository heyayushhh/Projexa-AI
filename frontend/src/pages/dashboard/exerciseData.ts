import data from "./exercise_db.json";

export default function Exercises() {
  return (
    <div>
      {Object.keys(data)
        .filter((section) => section !== "meta") // ✅ skip meta
        .map((section) => {
          const sectionData = data[section];

          return (
            <div key={section}>
              <h1>{section}</h1>

              {["beginner", "intermediate", "advanced"].map((level) => {
                const levelData = sectionData[level];

                // ✅ HANDLE BOTH ARRAY + OBJECT
                const items = Array.isArray(levelData)
                  ? levelData
                  : [levelData];

                return (
                  <div key={level}>
                    <h2>{level.toUpperCase()}</h2>

                    {items.map((item, index) => (
                      <div key={index} style={{ marginBottom: "20px" }}>
                        <h3>{item.name}</h3>

                        <p><strong>Duration:</strong> {item.duration}</p>
                        <p><strong>Instruction:</strong> {item.instruction}</p>

                        {/* 🔥 NEW FIELDS (your powerful data) */}
                        {item.goal && (
                          <p><strong>Goal:</strong> {item.goal}</p>
                        )}
                        {item.technique_cue && (
                          <p><strong>Technique:</strong> {item.technique_cue}</p>
                        )}
                        {item.self_check && (
                          <p><strong>Self Check:</strong> {item.self_check}</p>
                        )}

                        {/* Practice Material */}
                        <ul>
                          {item.practice_material.map((p, i) => (
                            <li key={i}>{p}</li>
                          ))}
                        </ul>
                      </div>
                    ))}
                  </div>
                );
              })}
            </div>
          );
        })}
    </div>
  );
}

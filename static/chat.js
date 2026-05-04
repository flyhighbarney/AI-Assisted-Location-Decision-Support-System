const chatMessages = document.getElementById("chatMessages");
const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");

const state = {
  step: "category",
  business_category: null,
  candidate_lat: null,
  candidate_lon: null,
  floor_area: null,
  last_result: null
};

addBotMessage(
  "Welcome. I will guide you through a store-location scenario for Worcester, MA. " +
  "First, what business category should we evaluate? For example: Restaurants, Grocery Stores, Fitness, or Retail."
);

sendBtn.addEventListener("click", handleSend);
chatInput.addEventListener("keydown", function (event) {
  if (event.key === "Enter") {
    handleSend();
  }
});

window.onMapLocationSelected = function (location) {
  state.candidate_lat = location.lat;
  state.candidate_lon = location.lon;

  if (state.step === "location") {
    addBotMessage(
      `Great, I captured the candidate location: ${location.lat.toFixed(6)}, ${location.lon.toFixed(6)}. ` +
      "Now enter the proposed store floor area in square meters."
    );
    state.step = "floor_area";
  }
};

async function handleSend() {
  const text = chatInput.value.trim();
  if (!text) return;

  addUserMessage(text);
  chatInput.value = "";

  try {
    if (state.step === "category") {
      const naicsCode = text.trim();
      
      if (!/^\d+$/.test(naicsCode)) {
        addBotMessage("Please enter a numeric NAICS code. For example: 4441.");
        return;
      }
    
      state.business_category = naicsCode;
      state.step = "location";
      
      addBotMessage(
        "Good. Now click the proposed store location on the map. " +
        "You can also type coordinates as: 42.24, -71.78"
      );
      return;
    }

    if (state.step === "location") {
      const coords = parseCoordinates(text);

      if (!coords) {
        addBotMessage("Please click the map or type coordinates in this format: 42.24, -71.78");
        return;
      }

      state.candidate_lat = coords.lat;
      state.candidate_lon = coords.lon;

      if (window.setCandidateLocation) {
        window.setCandidateLocation(coords.lat, coords.lon, false);
      }

      state.step = "floor_area";
      addBotMessage("Great. Now enter the proposed store floor area in square meters.");
      return;
    }

    if (state.step === "floor_area") {
      const area = Number(text.replace(/,/g, ""));

      if (!Number.isFinite(area) || area <= 0) {
        addBotMessage("Please enter a positive numeric floor area, such as 1000.");
        return;
      }

      state.floor_area = area;
      state.step = "ready";

      addBotMessage(
        `Thanks. I will run the Huff model for ${state.business_category}, ` +
        `location (${state.candidate_lat.toFixed(6)}, ${state.candidate_lon.toFixed(6)}), ` +
        `and floor area ${state.floor_area}.`
      );

      await runModel();
      return;
    }

    if (state.step === "ready") {
      await askQuestion(text);
      return;
    }
  } catch (error) {
    addErrorMessage(error.message || String(error));
  }
}

async function runModel() {
  addBotMessage("Running the model now...");

  const response = await fetch("/api/run_huff", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      candidate_lat: state.candidate_lat,
      candidate_lon: state.candidate_lon,
      business_category: state.business_category,
      floor_area: state.floor_area
    })
  });

  const data = await response.json();

  if (!response.ok || !data.ok) {
    throw new Error(data.error || "Model failed.");
  }

  state.last_result = data.result;

  renderResult(data.result);

  if (window.plotCompetitors) {
    window.plotCompetitors(data.result.competitors);
  }

  addBotMessage(
    data.explanation ||
    "Model completed. You can now ask follow-up questions about the result."
  );
}

async function askQuestion(question) {
  if (!state.last_result) {
    addBotMessage("Please complete a model run first.");
    return;
  }

  const response = await fetch("/api/ask", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      question,
      result: state.last_result
    })
  });

  const data = await response.json();

  if (!response.ok || !data.ok) {
    throw new Error(data.error || "The assistant could not answer.");
  }

  addBotMessage(data.answer);
}

function renderResult(result) {
  const summary = document.getElementById("resultSummary");
  const tableWrap = document.getElementById("competitorTable");

  summary.innerHTML = `
    <strong>Predicted Visits:</strong> ${escapeHtml(result.predicted_visits)}<br>
    <strong>Estimated Market Share:</strong> ${(Number(result.market_share) * 100).toFixed(2)}%<br>
    <strong>Runtime:</strong> ${escapeHtml(result.runtime_ms)} ms<br>
    <strong>Notes:</strong> ${escapeHtml(result.notes)}
  `;

  const competitors = Array.isArray(result.competitors) ? result.competitors : [];

  if (competitors.length === 0) {
    tableWrap.innerHTML = "No competitor records returned.";
    return;
  }

  tableWrap.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Name</th>
          <th>Distance</th>
          <th>Size</th>
          <th>Attraction</th>
        </tr>
      </thead>
      <tbody>
        ${competitors.map(c => `
          <tr>
            <td>${escapeHtml(c.name)}</td>
            <td>${escapeHtml(c.distance_miles)}</td>
            <td>${escapeHtml(c.size)}</td>
            <td>${escapeHtml(c.attraction)}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function parseCoordinates(text) {
  const parts = text.split(",").map(x => Number(x.trim()));

  if (parts.length !== 2) {
    return null;
  }

  const [lat, lon] = parts;

  if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
    return null;
  }

  return { lat, lon };
}

function addBotMessage(text) {
  addMessage(text, "bot");
}

function addUserMessage(text) {
  addMessage(text, "user");
}

function addErrorMessage(text) {
  addMessage(text, "error");
}

function addMessage(text, type) {
  const div = document.createElement("div");
  div.className = `message ${type}`;
  div.innerText = text;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

let candidateLocation = null;
let candidateMarker = null;
let competitorLayer = null;

const map = L.map("map").setView([42.2626, -71.8023], 12);

L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: "&copy; OpenStreetMap contributors"
}).addTo(map);

fetch("/static/data/worcester_cbgs_map.geojson")
  .then(response => {
    if (!response.ok) {
      throw new Error("GeoJSON not found");
    }
    return response.json();
  })
  .then(data => {
    const geoLayer = L.geoJSON(data, {
      style: {
        weight: 1,
        color: "#2563eb",
        opacity: 0.7,
        fillOpacity: 0.08
      }
    }).addTo(map);

    map.fitBounds(geoLayer.getBounds());
  })
  .catch(error => {
    console.warn("Worcester GeoJSON layer could not be loaded:", error);
  });

map.on("click", function (event) {
  setCandidateLocation(event.latlng.lat, event.latlng.lng, true);
});

function setCandidateLocation(lat, lon, notifyChat = false) {
  candidateLocation = {
    lat: Number(lat),
    lon: Number(lon)
  };

  if (candidateMarker) {
    candidateMarker.setLatLng([candidateLocation.lat, candidateLocation.lon]);
  } else {
    candidateMarker = L.marker([candidateLocation.lat, candidateLocation.lon])
      .addTo(map)
      .bindPopup("Proposed Store Location");
  }

  candidateMarker.openPopup();

  document.getElementById("selectedLocation").innerText =
    `Selected candidate location: ${candidateLocation.lat.toFixed(6)}, ${candidateLocation.lon.toFixed(6)}`;

  map.setView([candidateLocation.lat, candidateLocation.lon], 14);

  if (notifyChat && window.onMapLocationSelected) {
    window.onMapLocationSelected(candidateLocation);
  }
}

function getCandidateLocation() {
  return candidateLocation;
}

function plotCompetitors(competitors) {
  if (competitorLayer) {
    competitorLayer.remove();
  }

  competitorLayer = L.layerGroup().addTo(map);

  if (!Array.isArray(competitors)) {
    return;
  }

  competitors.forEach(comp => {
    if (comp.lat && comp.lon) {
      L.circleMarker([comp.lat, comp.lon], {
        radius: 6,
        weight: 1,
        color: "#dc2626",
        fillOpacity: 0.7
      })
        .addTo(competitorLayer)
        .bindPopup(
          `<strong>${escapeHtml(comp.name || "Competitor")}</strong><br>` +
          `Size: ${comp.size ?? "N/A"}<br>` +
          `Attraction: ${comp.attraction ?? "N/A"}`
        );
    }
  });
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

// Expose functions so chat.js can call them.
window.setCandidateLocation = setCandidateLocation;
window.getCandidateLocation = getCandidateLocation;
window.plotCompetitors = plotCompetitors;

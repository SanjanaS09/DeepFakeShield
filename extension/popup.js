document.getElementById("captureBtn").addEventListener("click", async () => {

    document.getElementById("resultSection").style.display = "block";
    document.getElementById("predictionText").innerText = "Analyzing...";
    document.getElementById("confidenceValue").innerText = "0%";

    const [tab] = await chrome.tabs.query({
        active: true,
        currentWindow: true
    });

    // Inject content script safely
    await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        files: ["content.js"]
    });

    // Capture screen
    chrome.tabs.captureVisibleTab(null, { format: "png" }, (dataUrl) => {

        chrome.tabs.sendMessage(tab.id, {
            action: "startCrop",
            screenshot: dataUrl
        });

    });

});

// Load existing result
document.addEventListener("DOMContentLoaded", loadResult);

// Listen for storage updates
chrome.storage.onChanged.addListener((changes, area) => {
    if (area === "local" && changes.lastResult) {
        displayResult(changes.lastResult.newValue);
    }
});

function loadResult() {
    chrome.storage.local.get("lastResult", (data) => {
        if (data.lastResult) {
            displayResult(data.lastResult);
        }
    });
}

function displayResult(data) {

    if (!data) return;

    document.getElementById("resultSection").style.display = "block";

    let realProbRaw = 0;
    let fakeProbRaw = 0;

    // ===== Use probabilities from backend =====
    if (data.fakeProbability !== undefined &&
        data.realProbability !== undefined) {

        fakeProbRaw = Number(data.fakeProbability) * 100;
        realProbRaw = Number(data.realProbability) * 100;

    } else if (data.confidence !== undefined) {

        const conf = Number(data.confidence) * 100;

        if (data.prediction === "FAKE") {
            fakeProbRaw = conf;
            realProbRaw = 100 - conf;
        } else {
            realProbRaw = conf;
            fakeProbRaw = 100 - conf;
        }
    }

    // ===== Calculate confidence from RAW values =====
    const confidencePercent = Math.max(realProbRaw, fakeProbRaw);

    // ===== Format for UI only =====
    const realDisplay = realProbRaw.toFixed(2);
    const fakeDisplay = fakeProbRaw.toFixed(2);
    const confidenceDisplay = confidencePercent.toFixed(2);

    document.getElementById("confidenceValue").textContent =
        confidenceDisplay + "%";

    document.getElementById("predictionText").innerText =
        data.prediction === "FAKE"
            ? "Fake Detected"
            : "Authentic Content";

    document.getElementById("labelValue").innerText = data.prediction;

    document.getElementById("realProb").innerText = realDisplay + "%";
    document.getElementById("fakeProb").innerText = fakeDisplay + "%";

    // ===== Progress Circle =====
    const progressCircle = document.getElementById("progressCircle");
    const radius = 50;
    const circumference = 2 * Math.PI * radius;

    progressCircle.style.strokeDasharray = circumference;

    const offset =
        circumference - (confidencePercent / 100) * circumference;

    progressCircle.style.strokeDashoffset = offset;

    progressCircle.style.stroke =
        data.prediction === "FAKE" ? "#f44336" : "#4caf50";

    // ===== Probability Bars =====
    document.getElementById("realBar").style.width = realDisplay + "%";
    document.getElementById("fakeBar").style.width = fakeDisplay + "%";

    // ===== Heatmap =====
    const heatmapImg = document.getElementById("heatmapImage");
    heatmapImg.src = "";

    if (data.heatmap) {
        heatmapImg.src = "data:image/png;base64," + data.heatmap;
    }
}


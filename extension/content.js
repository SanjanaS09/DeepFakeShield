// ================================
// DeepFake Shield - Crop & Detect
// ================================

let overlay = null;
let box = null;
let startX = 0;
let startY = 0;

// Listen for capture request from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {

    if (request.action === "startCrop") {

        // Prevent duplicate overlays
        if (overlay) return;

        const screenshot = request.screenshot;

        createOverlay();

        overlay.addEventListener("mousedown", (e) => {
            startX = e.clientX;
            startY = e.clientY;

            box = document.createElement("div");
            box.style.position = "absolute";
            box.style.border = "2px solid #00ffcc";
            box.style.background = "rgba(0,255,204,0.1)";
            box.style.left = startX + "px";
            box.style.top = startY + "px";

            overlay.appendChild(box);
        });

        overlay.addEventListener("mousemove", (e) => {
            if (!box) return;

            const width = e.clientX - startX;
            const height = e.clientY - startY;

            box.style.width = Math.abs(width) + "px";
            box.style.height = Math.abs(height) + "px";
            box.style.left = (width < 0 ? e.clientX : startX) + "px";
            box.style.top = (height < 0 ? e.clientY : startY) + "px";
        });

        overlay.addEventListener("mouseup", async () => {

            if (!box) return;

            const rect = box.getBoundingClientRect();

            if (rect.width < 20 || rect.height < 20) {
                cleanup();
                return;
            }

            showLoadingIndicator();

            const img = new Image();
            img.src = screenshot;

            img.onload = async () => {

                const canvas = document.createElement("canvas");
                const ctx = canvas.getContext("2d");

                canvas.width = rect.width;
                canvas.height = rect.height;

                ctx.drawImage(
                    img,
                    rect.left,
                    rect.top,
                    rect.width,
                    rect.height,
                    0,
                    0,
                    rect.width,
                    rect.height
                );

                const croppedDataUrl = canvas.toDataURL("image/png");

                const blob = await (await fetch(croppedDataUrl)).blob();
                const formData = new FormData();
                formData.append("file", blob, "capture.png");

                try {

                    const response = await fetch(
                        "http://localhost:5000/api/detection/image",
                        {
                            method: "POST",
                            body: formData
                        }
                    );

                    const data = await response.json();
                    removeLoader();

                    // Store result so popup can load it later
                    chrome.storage.local.set({
                        lastResult: {
                            prediction: data.prediction,
                            fakeProbability: data.fake_probability,
                            realProbability: data.real_probability,
                            confidence: data.confidence,
                            heatmap: data.xai && data.xai.heatmap ? data.xai.heatmap : null
                        }
                    }, () => {
                        console.log("Result saved to storage");
                    });

                    showFloatingResult(data);

                } catch (error) {
                    console.error("Detection error:", error);
                    alert("Backend not reachable.");
                }

                cleanup();
            };

        });

        overlay.addEventListener("keydown", (e) => {
            if (e.key === "Escape") cleanup();
        });
    }
});


// ================================
// Helper Functions
// ================================

function createOverlay() {

    overlay = document.createElement("div");
    overlay.style.position = "fixed";
    overlay.style.top = 0;
    overlay.style.left = 0;
    overlay.style.width = "100%";
    overlay.style.height = "100%";
    overlay.style.background = "rgba(0,0,0,0.3)";
    overlay.style.zIndex = 9999999;
    overlay.style.cursor = "crosshair";

    document.body.appendChild(overlay);
}

function cleanup() {
    if (overlay) {
        overlay.remove();
        overlay = null;
        box = null;
    }
}

function showLoadingIndicator() {

    // Remove any existing loader
    removeLoader();

    const loader = document.createElement("div");
    loader.id = "df-loading";
    loader.style.position = "fixed";
    loader.style.top = "20px";
    loader.style.right = "20px";
    loader.style.background = "#121212";
    loader.style.color = "#00ffcc";
    loader.style.padding = "10px 15px";
    loader.style.borderRadius = "6px";
    loader.style.zIndex = 99999999;
    loader.innerText = "Analyzing...";

    document.body.appendChild(loader);
}

function removeLoader() {
    const el = document.getElementById("df-loading");
    if (el) el.remove();
}

removeLoader();

function showFloatingResult(data) {

    const resultBox = document.createElement("div");

    resultBox.style.position = "fixed";
    resultBox.style.top = "60px";
    resultBox.style.right = "20px";
    resultBox.style.padding = "15px";
    resultBox.style.background = "#121212";
    resultBox.style.color = data.prediction === "FAKE" ? "#ff4444" : "#00ff88";
    resultBox.style.border = "2px solid";
    resultBox.style.borderColor =
        data.prediction === "FAKE" ? "#ff4444" : "#00ff88";
    resultBox.style.borderRadius = "8px";
    resultBox.style.zIndex = 99999999;
    resultBox.style.fontFamily = "Arial";
    resultBox.style.boxShadow = "0 0 15px rgba(0,0,0,0.5)";

    resultBox.innerHTML = `
        <strong>DeepFake Shield</strong><br><br>
        Prediction: ${data.prediction}<br>
        Confidence: ${(data.confidence * 100).toFixed(2)}%
    `;

    document.body.appendChild(resultBox);

    setTimeout(() => {
        resultBox.remove();
    }, 5000);
}
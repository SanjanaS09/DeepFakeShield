// chrome.runtime.onInstalled.addListener(() => {
//     chrome.contextMenus.create({
//         id: "deepfakeCheck",
//         title: "Check for Deepfake",
//         contexts: ["image"]
//     });
// });

// chrome.runtime.onInstalled.addListener(() => {
//     console.log("DeepFake Shield Installed");
// });

// chrome.contextMenus.onClicked.addListener((info, tab) => {

//     if (info.menuItemId === "deepfakeCheck") {

//         fetch("http://localhost:5000/image-url", {
//             method: "POST",
//             headers: { "Content-Type": "application/json" },
//             body: JSON.stringify({ image_url: info.srcUrl })
//         })
//             .then(res => res.json())
//             .then(data => {

//                 if (data.prediction === "FAKE") {
//                     chrome.tabs.sendMessage(tab.id, {
//                         action: "highlight",
//                         imageUrl: info.srcUrl
//                     });
//                 }

//                 chrome.notifications.create({
//                     type: "basic",
//                     iconUrl: "icon.png",
//                     title: "DeepFake Result",
//                     message: "Prediction: " + data.prediction +
//                         " | Confidence: " + data.confidence
//                 });
//             })
//             .catch(error => {
//                 alert("Backend not running.");
//             });
//     }
// });

chrome.runtime.onInstalled.addListener(() => {
    console.log("DeepFake Shield Installed");
});
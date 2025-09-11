// assets/folder_upload.js

// Function to initialize the folder upload
function initFolderUpload() {
    const container = document.getElementById("folder-upload-container");
    if (!container) return false; // container not yet rendered

    // Prevent double initialization
    if (container.dataset.initialized) return true;
    container.dataset.initialized = true;

    // Create hidden file input
    const input = document.createElement("input");
    input.type = "file";
    input.multiple = true;
    input.webkitdirectory = true;
    input.style.display = "none"; // hide the default input
    container.appendChild(input);

    // Create visible styled button
    const btn = document.createElement("button");
    btn.innerText = "Upload Folder";
    btn.style.padding = "0.5rem 1rem";
    btn.style.backgroundColor = "#00C8A0";
    btn.style.color = "white";
    btn.style.border = "none";
    btn.style.borderRadius = "8px";
    btn.style.cursor = "pointer";
    btn.style.fontWeight = "bold";
    btn.style.boxShadow = "0 4px 6px rgba(0, 0, 0, 0.2)";
    container.appendChild(btn);

    // Trigger file input when button is clicked
    btn.addEventListener("click", () => input.click());

    // Handle folder selection and upload
    input.addEventListener("change", async function () {
        const formData = new FormData();
        for (const file of input.files) {
            formData.append("files", file);
            formData.append(`paths[${file.name}]`, file.webkitRelativePath);
        }

        try {
            const resp = await fetch("https://l2c9dqkn.usw3.devtunnels.ms:8000/upload-folder", { method: "POST", body: formData });
            if (resp.ok) alert("Upload complete!");
            else alert("Upload failed!");
        } catch (err) {
            console.error(err);
            alert("Upload error!");
        }
    });

    return true;
}

// Repeatedly try to initialize until container exists
const interval = setInterval(() => {
    if (initFolderUpload()) clearInterval(interval);
}, 100);


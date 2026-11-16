
const audioInput = document.getElementById("audio-upload");
const fileName = document.getElementById("file-name");
const audioPreview = document.getElementById("audio-preview");
const transcriptInput = document.getElementById("transcript");
const languageInput = document.getElementById("language");
const uploadButton = document.getElementById("upload-button");
const statusElement = document.getElementById("status");
const resultsElement = document.getElementById("results");

let selectedFile = null;

audioInput.addEventListener("change", () => {
    selectedFile = audioInput.files[0];

    if (!selectedFile) return;

    fileName.textContent = selectedFile.nafselectedFile;
    audioPreview.style.display = "block";
});

uploadButton.addEventListener("click", async () => {
    if (!selectedFile) {
        statusElement.textContent = "Please select an audio file.";
        return;
    }


    const formData = new FormData();
    formData.append("audio", selectedFile);
    formData.append("transcript", transcriptInput.value.trim());
    formData.append("language", languageInput.value.trim());

    uploadButton.disabled = true;
    statusElement.textContent = "Uploading audio...";
    resultsElement.innerHTML = "";

    try {
        const response = await fetch("/api/upload/", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Upload failed.");
        }

        statusElement.textContent = "Upload successful. Processing...";
        await pollJob(data.job_id);

    } catch (error) {
        statusElement.textContent = `Error: ${error.message}`;
    } finally {
        uploadButton.disabled = false;
    }
});

async function pollJob(jobId) {
    while (true) {
        const response = await fetch(`/api/jobs/${jobId}/status/`);
        const job = await response.json();

        if (!response.ok) {
            throw new Error(job.error || "Could not check job status.");
        }

        if (job.status === "done") {
            statusElement.textContent = "Alignment completed!";
            await showResults(jobId);
            return;
        }

        if (job.status === "failed") {
            throw new Error(job.error || "Alignment failed.");
        }

        statusElement.textContent = `Processing... Status: ${job.status}`;
        await new Promise(resolve => setTimeout(resolve, 2000));
    }
}

async function showResults(jobId) {
    const response = await fetch(`/api/jobs/${jobId}/result/`);
    const data = await response.json();

    if (!response.ok) {
        throw new Error(data.error || "Could not retrieve results.");
    }

    const heading = document.createElement("h3");
    heading.textContent = "Aligned words";
    resultsElement.appendChild(heading);

    const list = document.createElement("ul");

    data.words.forEach(item => {
        const li = document.createElement("li");
        li.textContent = `${item.word}: ${item.start.toFixed(2)}s – ${item.end.toFixed(2)}s`;
        list.appendChild(li);
    });

    resultsElement.appendChild(list);

    const downloadLink = document.createElement("a");
    downloadLink.href = `/api/jobs/${jobId}/result/?format=textgrid`;
    downloadLink.textContent = "Download Praat TextGrid";
    downloadLink.download = `${jobId}.TextGrid`;

    resultsElement.appendChild(downloadLink);
}

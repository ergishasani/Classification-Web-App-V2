const form = document.getElementById('uploadForm');
const resultDiv = document.getElementById('result');

form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const fileInput = document.getElementById('fileInput');
    if (!fileInput.files.length) {
        alert('Please upload a file.');
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        const response = await fetch('http://127.0.0.1:5000/classify', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const response = await fetch('http://127.0.0.1:5000/classify', {
                method: 'POST',
                body: formData
            });
        }

        const data = await response.json();
        resultDiv.innerHTML = `
    <p><strong>File Name:</strong> ${data.filename}</p>
    <p><strong>Category:</strong> ${data.category}</p>
    <p><strong>Confidence:</strong> ${data.confidence.toFixed(2)}%</p>`;
    } catch (error) {
        resultDiv.innerHTML = `<p style="color: red;">${error.message}</p>`;
    }
});


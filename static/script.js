document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictionForm');
    const resultArea = document.getElementById('resultArea');
    const predictionResult = document.getElementById('predictionResult');
    const confidenceResult = document.getElementById('confidenceResult');
    const submitBtn = document.getElementById('submitBtn');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Visual feedback
        const originalBtnText = submitBtn.innerText;
        submitBtn.innerText = "ANALYZING...";
        submitBtn.disabled = true;
        resultArea.classList.add('hidden');

        // Gather data
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (response.ok) {
                // Display result
                predictionResult.innerText = result.prediction;
                confidenceResult.innerText = result.confidence;
                resultArea.classList.remove('hidden');
            } else {
                alert('Error: ' + result.error);
            }

        } catch (error) {
            console.error('Error:', error);
            alert('An unexpected error occurred.');
        } finally {
            submitBtn.innerText = originalBtnText;
            submitBtn.disabled = false;
        }
    });
});

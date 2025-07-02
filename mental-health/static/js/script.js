document.addEventListener('DOMContentLoaded', function() {
    // FAQ Form
    const faqForm = document.getElementById('faqForm');
    if (faqForm) {
        faqForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const question = document.getElementById('question').value.trim();
            const answerContainer = document.getElementById('answerContainer');
            const answerText = document.getElementById('answerText');
            
            if (question) {
                answerText.textContent = "Loading...";
                answerContainer.classList.remove('d-none');
                
                fetch('/faq', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: `question=${encodeURIComponent(question)}`
                })
                .then(response => response.json())
                .then(data => {
                    answerText.textContent = data.answer;
                })
                .catch(error => {
                    answerText.textContent = "Sorry, an error occurred. Please try again.";
                    console.error('Error:', error);
                });
            }
        });
    }
    
    // Stress Form
    const stressForm = document.getElementById('stressForm');
    if (stressForm) {
        stressForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const resultContainer = document.getElementById('resultContainer');
            const resultText = document.getElementById('resultText');
            const visualizationsDiv = document.getElementById('visualizations');
            
            // Get form data
            const formData = {
                gender: document.getElementById('gender').value,
                age: document.getElementById('age').value,
                sleep: document.getElementById('sleep').value,
                activity: document.getElementById('activity').value,
                heart_rate: document.getElementById('heart_rate').value,
                systolic: document.getElementById('systolic').value,
                diastolic: document.getElementById('diastolic').value
            };
            
            resultText.innerHTML = "<p>Analyzing your data...</p>";
            visualizationsDiv.innerHTML = "";
            resultContainer.classList.remove('d-none');
            
            fetch('/stress', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: new URLSearchParams(formData).toString()
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    resultText.innerHTML = `<p class="text-danger">${data.error}</p>`;
                } else {
                    // Display results
                    resultText.innerHTML = `<pre>${data.result}</pre>`;
                    
                    // Display visualizations if available
                    if (data.visualizations) {
                        visualizationsDiv.innerHTML = `
                            <h5 class="mt-4">Visualizations</h5>
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="visualization">
                                        <h6>Stress by Age Group</h6>
                                        <img src="data:image/png;base64,${data.visualizations.stress_vs_age}" class="img-fluid">
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="visualization">
                                        <h6>Feature Importance</h6>
                                        <img src="data:image/png;base64,${data.visualizations.feature_importance}" class="img-fluid">
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="visualization">
                                        <h6>Confusion Matrix</h6>
                                        <img src="data:image/png;base64,${data.visualizations.confusion_matrix}" class="img-fluid">
                                    </div>
                                </div>
                            </div>
                        `;
                    }
                }
            })
            .catch(error => {
                resultText.innerHTML = `<p class="text-danger">An error occurred: ${error.message}</p>`;
                console.error('Error:', error);
            });
        });
    }
    
    // Load visualizations on the visualizations page
    if (document.getElementById('stressAgeImg')) {
        fetch('/stress', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: 'dummy=true'  // Just to trigger the visualization generation
        })
        .then(response => response.json())
        .then(data => {
            if (data.visualizations) {
                document.getElementById('stressAgeImg').src = `data:image/png;base64,${data.visualizations.stress_vs_age}`;
                document.getElementById('featureImportanceImg').src = `data:image/png;base64,${data.visualizations.feature_importance}`;
                document.getElementById('confusionMatrixImg').src = `data:image/png;base64,${data.visualizations.confusion_matrix}`;
            }
        });
    }
});
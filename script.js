document.getElementById('generate-button').addEventListener('click', function() {
    fetch('http://127.0.0.1:5000/generate-face')
    .then(response => response.blob())
    .then(blob => {
        const imageUrl = URL.createObjectURL(blob);
        document.getElementById('generated-face').src = imageUrl;
    })
    .catch(error => console.error('Error:', error));
});

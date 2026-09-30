async function updateStatus() {
  try {
    const response = await fetch('/status');
    const data = await response.json();
    const statusEl = document.getElementById('status');

    statusEl.textContent = "Status: " + data.status;

    // Change color based on status
    if (data.status === "Attentive") {
      statusEl.className = "good";
    } else if (data.status === "INATTENTIVE") {
      statusEl.className = "warn";
    } else if (data.status === "DROWSY") {
      statusEl.className = "bad";
    }
  } catch (err) {
    console.error(err);
  }
}

setInterval(updateStatus, 1000);
// Handle exam submission
document.getElementById('examForm').addEventListener('submit', async function(e) {
  e.preventDefault(); // prevent page reload

  const formData = new FormData(this);
  const data = Object.fromEntries(formData);

  const response = await fetch('/submit_exam', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });

  const result = await response.json();
  document.getElementById('resultMsg').textContent = result.message;
});

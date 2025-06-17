async function submitForm(endpoint) {
  const form = document.getElementById('creditForm');
  const formData = new FormData(form);
  const payload = {};
  formData.forEach((val, key) => {
    payload[key] = val;
  });

  document.getElementById('result').innerText = 'Loading...';

  try {
    const res = await fetch(`/${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    document.getElementById('result').innerText = data.prediction || 'No prediction returned.';
  } catch (err) {
    document.getElementById('result').innerText = 'Error contacting server.';
  }
}

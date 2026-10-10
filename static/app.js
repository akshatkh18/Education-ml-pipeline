const form = document.getElementById('prediction-form');
const button = document.getElementById('predict-button');
form.addEventListener('submit', () => {
  button.disabled = true;
  button.textContent = 'Estimating your score…';
});
window.addEventListener('pageshow', () => {
  button.disabled = false;
  button.textContent = 'Estimate mathematics score ↗';
});
const errors = document.getElementById('errors');
if (errors) errors.focus();

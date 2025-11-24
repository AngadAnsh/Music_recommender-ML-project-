const input = document.getElementById('songInput');
const suggestionsBox = document.getElementById('suggestions');

let timer = null;

input.addEventListener('input', () => {
  clearTimeout(timer);
  const query = input.value.trim();

  if (!query) {
    suggestionsBox.style.display = 'none';
    return;
  }

  // Delay (debounce)
  timer = setTimeout(() => fetchSuggestions(query), 200);
});

function fetchSuggestions(query) {
  fetch(`/suggest?q=${encodeURIComponent(query)}`)
    .then(res => res.json())
    .then(suggestions => {
      if (!suggestions.length) {
        suggestionsBox.style.display = 'none';
        return;
      }

      suggestionsBox.innerHTML = suggestions
        .map(item => `<div class="item" onclick="selectSuggestion('${item}')">${item}</div>`)
        .join('');

      suggestionsBox.style.display = 'block';
    })
    .catch(() => {
      suggestionsBox.style.display = 'none';
    });
}

function selectSuggestion(name) {
  document.getElementById('songInput').value = name;
  suggestionsBox.style.display = 'none';
}

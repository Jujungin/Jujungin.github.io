const params = new URLSearchParams(window.location.search);
const file = params.get("file");

if (file) {
  fetch(`source/${file}`)
    .then((res) => res.text())
    .then((md) => {
      document.getElementById("content").innerHTML = marked.parse(md);

      hljs.highlightAll();

      generateTOC();
      addCopyButtons();
    });
}

function generateTOC() {
  const headers = document.querySelectorAll(
    "#content h1, #content h2, #content h3",
  );
  const toc = document.getElementById("toc-list");

  headers.forEach((header, i) => {
    const id = "section-" + i;
    header.id = id;

    const link = document.createElement("a");
    link.href = "#" + id;
    link.textContent = header.textContent;

    toc.appendChild(link);
  });
}

function addCopyButtons() {
  document.querySelectorAll("pre").forEach((block) => {
    const btn = document.createElement("button");
    btn.innerText = "copy";
    btn.className = "copy-btn";

    btn.onclick = () => {
      navigator.clipboard.writeText(block.innerText);
      btn.innerText = "copied!";
      setTimeout(() => (btn.innerText = "copy"), 1000);
    };

    block.appendChild(btn);
  });
}

// MathJax configuration for pymdownx.arithmatex in "generic" mode.
//
// The readthedocs theme serves ordinary full page loads, so MathJax typesets
// once on load and nothing further is needed. (Material for MkDocs uses
// instant navigation and would additionally need a `document$` subscription
// to re-typeset after each client-side page swap.)
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

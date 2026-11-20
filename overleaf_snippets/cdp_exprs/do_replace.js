(() => {
  const newSection = window.__newSection;
  if (!newSection || newSection.length < 100) return {ok:false, reason:'empty', len: (newSection||'').length};
  let view = null;
  for (const el of document.querySelectorAll('*')) {
    for (const k of Object.keys(el)) {
      try {
        const v = el[k];
        if (v && v.state && v.state.doc && typeof v.dispatch === 'function' && v.state.doc.toString) { view = v; break; }
      } catch (e) {}
    }
    if (view) break;
  }
  if (!view) return {ok:false, reason:'no cm'};
  const doc = view.state.doc.toString();
  const startMark = '\\section{Model II: Measures of Music Similarity}\\label{sec:model-similarity}';
  const endMark = '\\section{Model III: Genre Comparison, Evolution and Revolutions}\\label{sec:model-genre}';
  const i0 = doc.indexOf(startMark);
  const i1 = doc.indexOf(endMark);
  if (i0 < 0 || i1 < 0) return {ok:false, reason:'markers', i0, i1};
  view.dispatch({changes:{from:i0, to:i1, insert:newSection}});
  const d2 = view.state.doc.toString();
  return {ok:true, len:newSection.length, hasPCA:d2.includes('PCA then Euclidean'), linePreview:d2.slice(i0, i0+100)};
})()
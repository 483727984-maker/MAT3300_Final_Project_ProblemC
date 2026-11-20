(() => {
  const text = atob(window.__b64);
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
  const doc = view.state.doc;
  view.dispatch({changes:{from:0, to:doc.length, insert:text}});
  const out = view.state.doc.toString();
  return {ok:true, len:out.length, head:out.slice(0,100), hasPCA: out.includes('PCA then Euclidean')};
})()
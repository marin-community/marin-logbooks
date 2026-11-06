"""Key results and trajectory sections of the overnight page."""
import html
import json

ORDER = ("keep166", "global188", "global200", "log32768", "log65536sq", "log65536a20", "keep166-raw")
CAND = ("keep166", "global188", "global200", "log32768")
DASH = {"keep166": "", "global188": "7 4", "global200": "", "log32768": "2 3"}
COLOR = {"keep166": "var(--c1)", "global188": "var(--c2)", "global200": "var(--c3)", "log32768": "var(--c4)"}
SHORT = {"keep166": "Keep 1.66", "keep166-raw": "Keep 1.66, raw", "log65536sq": "Log 65536, squared (queue 64)",
         "global188": "Global 1.88", "log32768": "Log 32768", "log65536a20": "Log 65536, slope 0.2", "global200": "Global 2.00"}
CODE = {"keep166": "No", "keep166-raw": "No", "global188": "No", "global200": "No",
        "log32768": "Export: serving support",
        "log65536sq": "Train: power field. Export: serving support",
        "log65536a20": "Train: slope field. Export: serving support"}
FAR, N8, CROSS = "distance_131073_plus", "8needle", "8needle/distance_131073_plus"
STEPS = [str(s) for s in range(390251, 390351)]
JIT_LO, JIT_HI = 0.0002, 0.0007
CE_LO, CE_HI = 1.2e-4, 1.5e-4
CE = "train/cross_entropy_loss"

CSS = """
:root{--c1:#2a78d6;--c2:#eb6834;--c3:#1baf7a;--c4:#9a5fd0;--bandhi:rgba(91,103,112,.24);--bandlo:rgba(91,103,112,.13)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--c1:#3987e5;--c2:#d95926;--c3:#199e70;--c4:#a678dc;--bandhi:rgba(154,166,175,.28);--bandlo:rgba(154,166,175,.14)}}
:root[data-theme="dark"]{--c1:#3987e5;--c2:#d95926;--c3:#199e70;--c4:#a678dc;--bandhi:rgba(154,166,175,.28);--bandlo:rgba(154,166,175,.14)}
h3{font-size:15px;font-weight:600;margin:14px 0 0}
.kpanels{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:14px}
.kp{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:12px 14px 10px;min-width:0;container-type:inline-size}
.kp figcaption{margin-bottom:8px}
.krow{display:grid;grid-template-columns:190px minmax(0,1fr);align-items:center;column-gap:10px;border-radius:4px;outline-offset:1px;cursor:default;min-height:30px}
.krow:hover,.krow:focus-visible{background:color-mix(in srgb,var(--line) 45%,transparent)}
.krow.axisrow{cursor:auto;background:none;min-height:0}
.klab{font-size:13px;line-height:1.25;padding-left:4px}
.ktrack{position:relative;height:30px}
.kgrid{position:absolute;top:0;bottom:0;width:1px;background:var(--grid)}
.kgrid.zero{width:2px;margin-left:-.5px;background:var(--muted)}
.kband{position:absolute;top:0;bottom:0}
.kband.hi{background:var(--bandhi)}.kband.lo{background:var(--bandlo)}
.kci{position:absolute;top:14px;height:2px;border-radius:1px}
.kdot{position:absolute;top:10px;width:10px;height:10px;margin-left:-5px;border-radius:50%;box-shadow:0 0 0 2px var(--panel)}
.kaxis{position:relative;height:20px}
.ktick{position:absolute;top:4px;transform:translateX(-50%);font:12px var(--mono);color:var(--muted);white-space:nowrap}
@container (max-width:460px){.krow{grid-template-columns:minmax(0,1fr)}.klab{padding-top:5px}.ktrack{height:22px}.kci{top:10px}.kdot{top:6px}.krow.axisrow .klab{display:none}}
#ktip{position:fixed;z-index:10;display:none;max-width:min(300px,calc(100vw - 24px));padding:7px 10px;background:var(--panel);color:var(--ink);border:1px solid var(--muted);border-radius:6px;font-size:13px;line-height:1.4;pointer-events:none;box-shadow:0 2px 8px rgba(0,0,0,.25)}
#ktip b{font-weight:600;display:block}
#ktip span{font-family:var(--mono);font-size:12px;display:block}
#ktip i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}
.ktab table{min-width:1040px}
.ktab2 table{min-width:1020px}
.ktab thead th,.ktab2 thead th{white-space:normal;vertical-align:bottom}
.ktab tbody th,.ktab2 tbody th{white-space:normal;min-width:112px;max-width:170px}
td.kcode{font-family:var(--sans);white-space:normal;min-width:170px;text-align:left}
.lc{min-width:0}
.lc svg{display:block;width:100%;overflow:visible;touch-action:pan-y}
.lleg{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:13px;margin-bottom:6px}
.lleg span{display:inline-flex;align-items:center;gap:6px}
.lleg svg{width:22px;height:10px;flex:none}
.lt{font:12px var(--mono);fill:var(--muted)}
.ld{font:12px var(--sans);fill:var(--ink)}
"""

JS = r"""<div id="ktip" role="status"></div><script>
(function(){
  var tip=document.getElementById('ktip'),on=null;
  function place(x,y){var r=tip.getBoundingClientRect(),m=12,l=x+m,t=y+m;
    if(l+r.width>innerWidth-m)l=Math.max(m,x-r.width-m);
    if(t+r.height>innerHeight-m)t=Math.max(m,y-r.height-m);
    tip.style.left=l+'px';tip.style.top=t+'px'}
  function show(h,x,y,el){tip.innerHTML=h;tip.style.display='block';place(x,y);on=el}
  function hide(){tip.style.display='none';on=null}
  document.querySelectorAll('.krow[data-val]').forEach(function(el){
    var h=function(){return '<b>'+el.dataset.name+'</b><span>'+el.dataset.val+'</span>'};
    el.addEventListener('pointermove',function(e){if(e.pointerType!=='touch')show(h(),e.clientX,e.clientY,el)});
    el.addEventListener('pointerleave',function(e){if(e.pointerType!=='touch')hide()});
    el.addEventListener('click',function(e){e.stopPropagation();if(on===el)hide();else show(h(),e.clientX,e.clientY,el)});
    el.addEventListener('focus',function(){var r=el.getBoundingClientRect();show(h(),r.left+8,r.bottom,el)});
    el.addEventListener('blur',hide)});
  document.addEventListener('click',hide);
  addEventListener('scroll',hide,{passive:true});
  document.addEventListener('keydown',function(e){if(e.key==='Escape')hide()});
  var NS='http://www.w3.org/2000/svg';
  function E(n,a,p){var e=document.createElementNS(NS,n);for(var k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e}
  function fmt(v,s){return s.signed&&v===0?'0':(s.signed&&v>=0?'+':'')+v.toFixed(s.digits)}
  function draw(box,s){
    box.textContent='';var w=box.clientWidth;if(!w)return;
    if(s.series.length>1){var lg=document.createElement('div');lg.className='lleg';s.series.forEach(function(q){
      var sp=document.createElement('span'),k=E('svg',{viewBox:'0 0 22 10'});
      if(q.line!==false)E('line',{x1:0,x2:22,y1:5,y2:5,stroke:q.color,'stroke-width':2,'stroke-dasharray':q.dash||''},k);
      E('circle',{cx:11,cy:5,r:q.line===false?4:0,fill:q.color},k);sp.appendChild(k);sp.appendChild(document.createTextNode(q.name));lg.appendChild(sp)});box.appendChild(lg)}
    var lab=s.endlabels&&w>=600,M={l:s.ml||58,r:lab?92:12,t:6,b:24},H=s.h||220;
    var svg=E('svg',{width:w,height:H,viewBox:'0 0 '+w+' '+H,tabindex:0,role:'img','aria-label':s.alt},box);
    var X=function(v){return M.l+(v-s.xmin)/(s.xmax-s.xmin)*(w-M.l-M.r)};
    var T=s.ylog?function(v){return Math.log10(v)}:function(v){return v};
    var Y=function(v){var c=Math.min(s.ymax,Math.max(s.ymin,v));return H-M.b-(T(c)-T(s.ymin))/(T(s.ymax)-T(s.ymin))*(H-M.t-M.b)};
    (s.bands||[]).forEach(function(b){E('rect',{x:M.l,width:w-M.l-M.r,y:Y(b.y1),height:Y(b.y0)-Y(b.y1),fill:b.fill},svg)});
    (s.xbands||[]).forEach(function(b){E('rect',{x:X(b.x)-b.w/2,width:b.w,y:Y(b.y1),height:Y(b.y0)-Y(b.y1),fill:b.fill},svg)});
    s.yticks.forEach(function(v){var y=Y(v);E('line',{x1:M.l,x2:w-M.r,y1:y,y2:y,stroke:v===s.zero?'var(--muted)':'var(--grid)','stroke-width':v===s.zero?1.5:1},svg);
      E('text',{x:M.l-6,y:y+4,'text-anchor':'end',class:'lt'},svg).textContent=s.ylabels?s.ylabels[String(v)]:fmt(v,s.ytick||s)});
    s.xticks.forEach(function(v){var t=E('text',{x:X(v),y:H-7,'text-anchor':'middle',class:'lt'},svg);t.textContent=v});
    (s.hlines||[]).forEach(function(l){var y=Y(l.y);E('line',{x1:M.l,x2:w-M.r,y1:y,y2:y,stroke:'var(--muted)','stroke-width':1.5,'stroke-dasharray':'3 3'},svg);
      E('text',{x:w-M.r-4,y:y-4,'text-anchor':'end',class:'lt'},svg).textContent=l.label});
    s.series.forEach(function(q){
      if(q.line!==false){var d=q.pts.map(function(p,i){return(i?'L':'M')+X(p[0]).toFixed(1)+' '+Y(p[1]).toFixed(1)}).join('');
        E('path',{d:d,fill:'none',stroke:q.color,'stroke-width':s.thin?1.5:2,'stroke-linejoin':'round','stroke-dasharray':q.dash||''},svg)}
      if(q.dots)q.pts.forEach(function(p){E('circle',{cx:X(p[0])+(q.dx||0),cy:Y(p[1]),r:q.r||4.5,fill:q.color,stroke:'var(--panel)','stroke-width':2},svg)})});
    if(lab){var ls=s.series.filter(function(q){return q.line!==false}).map(function(q){var p=q.pts[q.pts.length-1];return{q:q,y:Y(p[1])}}).sort(function(a,b){return a.y-b.y});
      for(var i=1;i<ls.length;i++)if(ls[i].y-ls[i-1].y<14)ls[i].y=ls[i-1].y+14;
      ls.forEach(function(o){E('text',{x:w-M.r+6,y:o.y+4,class:'ld'},svg).textContent=o.q.name})}
    var xs=[];s.series.forEach(function(q){q.pts.forEach(function(p){if(xs.indexOf(p[0])<0)xs.push(p[0])})});xs.sort(function(a,b){return a-b});
    var cross=E('line',{y1:M.t,y2:H-M.b,stroke:'var(--muted)','stroke-width':1,opacity:0},svg);
    function at(x,cx,cy){cross.setAttribute('x1',X(x));cross.setAttribute('x2',X(x));cross.setAttribute('opacity',1);
      var h='<b>'+(s.xname||'Step')+' '+x+'</b>';s.series.forEach(function(q){q.pts.forEach(function(p){if(p[0]===x)h+='<span><i style="background:'+q.color+'"></i>'+q.name+': '+fmt(p[1],s)+(p[2]?' '+p[2]:'')+'</span>'})});
      show(h,cx,cy,svg)}
    function near(cx){var r=svg.getBoundingClientRect(),v=s.xmin+(cx-r.left-M.l)/(w-M.l-M.r)*(s.xmax-s.xmin),b=xs[0];xs.forEach(function(x){if(Math.abs(x-v)<Math.abs(b-v))b=x});return b}
    svg.addEventListener('pointermove',function(e){at(near(e.clientX),e.clientX,e.clientY)});
    svg.addEventListener('pointerleave',function(){cross.setAttribute('opacity',0);hide()});
    svg.addEventListener('click',function(e){e.stopPropagation();at(near(e.clientX),e.clientX,e.clientY)});
    var cur=xs.length-1;svg.addEventListener('keydown',function(e){if(e.key!=='ArrowLeft'&&e.key!=='ArrowRight')return;e.preventDefault();cur=Math.max(0,Math.min(xs.length-1,cur+(e.key==='ArrowLeft'?-1:1)));var r=svg.getBoundingClientRect();at(xs[cur],r.left+X(xs[cur]),r.top+20)});
    svg.addEventListener('focus',function(){var r=svg.getBoundingClientRect();at(xs[cur],r.left+X(xs[cur]),r.top+20)});
    svg.addEventListener('blur',function(){cross.setAttribute('opacity',0);hide()});
  }
  document.querySelectorAll('.lc').forEach(function(box){
    var s=JSON.parse(box.dataset.spec),last=0;draw(box,s);
    new ResizeObserver(function(){if(box.clientWidth!==last){last=box.clientWidth;draw(box,s)}}).observe(box)});
})();
</script>"""


def _pct(v, lo, hi):
    assert lo <= v <= hi, (v, lo, hi)
    return "%.3f%%" % (100 * (v - lo) / (hi - lo))


def _iv(v):
    return "%+.4f [%+.4f, %+.4f]" % tuple(v)


def _cell(v):
    return "%+.4f<small>[%+.4f, %+.4f]</small>" % tuple(v)


def _val(x, ci, d="%.4f"):
    return (d + "<small>[" + d + ", " + d + "]</small>") % (x, ci[0], ci[1])


def _color(a):
    return COLOR.get(a, "var(--muted)")


def panel(title, rows, lo, hi, ticks, tickfmt, zero=0.0, bands=()):
    """A dot-and-interval panel. Each row is (arm, value, low or None, high or None, tooltip text)."""
    def grid():
        out = "".join('<i class="kband %s" style="left:%s;width:calc(%s - %s)"></i>' % (c, _pct(a, lo, hi), _pct(b, lo, hi), _pct(a, lo, hi)) for a, b, c in bands)
        return out + "".join('<i class="kgrid%s" style="left:%s"></i>' % (" zero" if x == zero else "", _pct(x, lo, hi)) for x in ticks)
    body = []
    for a, v, l, h, tip in rows:
        name = html.escape(SHORT[a]); c = _color(a)
        ci = "" if l is None else '<i class="kci" style="left:%s;width:calc(%s - %s);background:%s"></i>' % (_pct(l, lo, hi), _pct(h, lo, hi), _pct(l, lo, hi), c)
        body.append('<div class="krow" tabindex="0" data-name="%s: %s" data-val="%s" aria-label="%s, %s: %s"><div class="klab">%s</div><div class="ktrack">%s%s<i class="kdot" style="left:%s;background:%s"></i></div></div>'
                    % (name, html.escape(title), tip, name, html.escape(title), tip, name, grid(), ci, _pct(v, lo, hi), c))
    axis = "".join('<span class="ktick" style="left:%s">%s</span>' % (_pct(x, lo, hi), tickfmt(x)) for x in ticks)
    return '<figure class="kp"><figcaption>%s</figcaption>%s<div class="krow axisrow"><div class="klab"></div><div class="kaxis">%s</div></div></figure>' % (html.escape(title), "".join(body), axis)


def _signed(x):
    return "0" if x == 0 else "%+.3f" % x


def _lc(title, spec, cap=None):
    spec = json.dumps(spec, separators=(",", ":"))
    note = '<p class="note" style="margin-top:6px">%s</p>' % cap if cap else ""
    return '<figure class="kp"><figcaption>%s</figcaption><div class="lc" data-spec="%s"></div>%s</figure>' % (html.escape(title), html.escape(spec, quote=True), note)


def _ser(a, pts, **kw):
    return {"name": SHORT[a], "color": COLOR[a], "dash": DASH[a], "pts": pts, **kw}


def clean_panels(RES):
    K = RES["arms"]["keep166"]["eval"]; out = []
    bands = ((-JIT_HI, JIT_HI, "lo"), (-JIT_LO, JIT_LO, "hi"))
    for key, n in (("c4096", "4K"), ("c65536", "65K"), ("c262144", "262K")):
        rows = [(a, RES["arms"][a]["eval"][key] - K[key], None, None, "%.4f (%+.4f from Keep 1.66)" % (RES["arms"][a]["eval"][key], RES["arms"][a]["eval"][key] - K[key])) for a in ORDER]
        out.append(panel("Clean macro, " + n, rows, -0.0014, 0.0024, (-0.001, 0.0, 0.001, 0.002), _signed, bands=bands))
    return '<div class="kpanels">' + "".join(out) + "</div>"


def nll_panel(RES, metric, title):
    P = RES["pairs"]; rows = []
    for a in ORDER:
        if a in ("keep166", "keep166-raw"): continue
        m, l, h = P[a + "-keep166|" + metric]
        rows.append((a, m, l, h, _iv((m, l, h))))
    return panel(title, rows, -0.0035, 0.0015, (-0.003, -0.002, -0.001, 0.0, 0.001), _signed)


def hard_panels(HARD):
    out = []
    cells = ((FAR, "farthest band"), (N8, "8 needles"), (CROSS, "8 needles, farthest band"))
    for metric, name, lo, hi, ticks in (("full", "Full NLL change", -0.0105, 0.0045, (-0.009, -0.006, -0.003, 0.0, 0.003)), ("rec", "Recovery change", -0.0035, 0.0145, (0.0, 0.004, 0.008, 0.012))):
        for g, lab in cells:
            c = HARD["groups"][g]; rows = []
            for a in ORDER:
                if a in ("keep166", "keep166-raw"): continue
                v = c["delta"][a][metric]; rows.append((a, v[0], v[1], v[2], _iv(v)))
            out.append(panel("%s, %s (%d sources)" % (name, lab, c["n"]), rows, lo, hi, ticks, _signed))
    return '<div class="kpanels">' + "".join(out) + "</div>"


def hard_table(HARD, table):
    rows = []
    gs = [HARD["groups"][g] for g in (FAR, N8, CROSS)]
    for a in ORDER:
        r = [SHORT[a]]
        for c in gs:
            x = c["arms"][a]
            r += [_val(x["rec"], x["rec_ci"], "%.3f"), _val(x["full"], x["full_ci"])]
        rows.append(tuple(r))
    head = ["Recipe"]
    for c, n in zip(gs, ("Farthest band", "8 needles", "8 needles, farthest")):
        head += ["%s (%d), recovery" % (n, c["n"]), "%s (%d), full NLL" % (n, c["n"])]
    return table(head, rows).replace('<div class="tw">', '<div class="tw ktab2">', 1)


def all_cells(HARD, table):
    names = lambda g: g.replace("distance_0_32768", "0 to 32K").replace("distance_32769_65536", "32K to 65K").replace("distance_65537_131072", "65K to 131K").replace("distance_131073_plus", "over 131K").replace("needle", " needles").replace("/", ", ")
    rows = {"rec": [], "full": []}
    for g, c in HARD["groups"].items():
        for m in rows:
            r = ["%s (%d%s)" % ("All" if g == "all" else names(g), c["n"], ", too few to read" if c["n"] < 25 else "")]
            for a in ORDER:
                x = c["arms"][a]
                r.append(("%.3f" if m == "rec" else "%.4f") % x[m])
            rows[m].append(tuple(r))
    head = ["Group (sources)"] + [SHORT[a] for a in ORDER]
    return ('<details><summary>Every distance and needle group: recovery</summary>' + table(head, rows["rec"]) + '</details>'
            '<details><summary>Every distance and needle group: full NLL</summary>' + table(head, rows["full"]) + '</details>')


def summary_table(A, RES, HARD, GRAD, table):
    P = RES["pairs"]; rows = []
    for n in ORDER:
        o = RES["arms"][n]; e = o["eval"]; g = GRAD[n]["train/grad_norm"]; c = GRAD[n]["train/grad_clipped"]; ks = STEPS
        assert len(g) == 100 and len(c) == 100, n
        early = max(g[k] for k in ks[:20]); cf = int(sum(c[k] for k in ks[:20])); ca = int(sum(c.values()))
        if n == "keep166":
            fu = fa = n8 = "Reference"
        else:
            fu = _cell(P[n + "-keep166|full_context"]); fa = _cell(HARD["groups"][FAR]["delta"][n]["full"]); n8 = _cell(HARD["groups"][N8]["delta"][n]["full"])
        rows.append((SHORT[n], "%.4f" % e["c262144"], "%.3f<small>[%.3f, %.3f]</small>" % (e["rec"], *e["rec_ci"]), fu, fa, n8, "%.2f" % early, "%d / %d" % (cf, ca), CODE[n]))
    tab = table(["Recipe", "Clean 262K", "Recovery [95%]", "Full NLL change, all [95%]", "Full NLL change, farthest band (92) [95%]", "Full NLL change, 8 needles (98) [95%]", "Largest grad norm, first 20 steps", "Steps clipped, first 20 / all 100", "Extra code to train or export"], rows)
    for c in set(CODE.values()):
        tab = tab.replace("<td>%s</td></tr>" % c, '<td class="kcode">%s</td></tr>' % c)
    return tab.replace('<div class="tw">', '<div class="tw ktab">', 1)


def grad_chart(GRAD, n, h):
    xs = STEPS[:n]
    last = int(xs[-1])
    return {"alt": "Grad norm per step on a log scale for the four candidates, with the clip threshold at 1.0.", "h": h, "ylog": True, "ymin": 0.1, "ymax": 30, "yticks": [0.1, 1, 10],
            "ylabels": {"0.1": "0.1", "1": "1", "10": "10"}, "digits": 3, "xmin": 390251, "xmax": last, "xticks": ([390251, 390255, 390260, 390265, 390270] if n == 20 else [390251, 390275, 390300, 390325, 390350]),
            "hlines": [{"y": 1.0, "label": "clip threshold 1.0"}], "endlabels": False, "thin": n > 20,
            "series": [_ser(a, [[int(s), GRAD[a]["train/grad_norm"][s]] for s in xs], dots=n == 20, r=3) for a in CAND]}


def traj_ce(RES):
    S = lambda a: RES["arms"][a]["series"][CE]
    raw = {"alt": "Train cross-entropy per step for the four candidates.", "h": 220, "ymin": 0.74, "ymax": 1.0, "yticks": [0.8, 0.9, 1.0], "digits": 4, "ytick": {"digits": 1}, "xmin": 390251, "xmax": 390350, "xticks": [390251, 390275, 390300, 390325, 390350], "thin": True,
           "series": [_ser(a, [[int(s), S(a)[s]] for s in STEPS]) for a in CAND]}
    diff = {"alt": "Train cross-entropy per step minus Keep 1.66 merged, with the one-rack repeat band.", "h": 240, "ymin": -0.005, "ymax": 0.001, "yticks": [-0.004, -0.003, -0.002, -0.001, 0, 0.001], "zero": 0, "signed": True, "digits": 5, "ytick": {"digits": 3, "signed": True}, "xmin": 390251, "xmax": 390350, "xticks": [390251, 390275, 390300, 390325, 390350], "endlabels": False, "thin": True,
            "bands": [{"y0": -CE_HI, "y1": CE_HI, "fill": "var(--bandlo)"}, {"y0": -CE_LO, "y1": CE_LO, "fill": "var(--bandhi)"}],
            "series": [_ser(a, [[int(s), S(a)[s] - S("keep166")[s]] for s in STEPS]) for a in CAND if a != "keep166"]}
    zoom = dict(diff, alt="Train cross-entropy minus Keep 1.66 merged, steps 390271 to 390350, zoomed to the band.", ymin=-0.0012, ymax=0.0004, yticks=[-0.0012, -0.0008, -0.0004, 0, 0.0004], ytick={"digits": 4, "signed": True}, xmin=390271, xticks=[390271, 390290, 390310, 390330, 390350],
                series=[dict(q, pts=[p for p in q["pts"] if p[0] >= 390271]) for q in diff["series"]])
    return raw, diff, zoom


def traj_eval(RES, IN):
    dx = {"keep166": -9, "global188": -3, "global200": 3, "log32768": 9}
    Kb, Kf = RES["arms"]["keep166"]["base"]["c262144"], RES["arms"]["keep166"]["eval"]["c262144"]
    off = {"alt": "Offline clean macro at 262K at steps 390251 and 390351, not joined.", "h": 230, "ymin": 1.642, "ymax": 1.652, "yticks": [1.642, 1.644, 1.646, 1.648, 1.65, 1.652], "digits": 4, "ytick": {"digits": 3}, "xmin": 390241, "xmax": 390361, "xticks": [390251, 390351],
           "xbands": [{"x": x, "w": 44, "y0": k - JIT_HI, "y1": k + JIT_HI, "fill": "var(--bandlo)"} for x, k in ((390251, Kb), (390351, Kf))] + [{"x": x, "w": 44, "y0": k - JIT_LO, "y1": k + JIT_LO, "fill": "var(--bandhi)"} for x, k in ((390251, Kb), (390351, Kf))],
           "series": [_ser(a, [[390251, RES["arms"][a]["base"]["c262144"]], [390351, RES["arms"][a]["eval"]["c262144"]]], line=False, dots=True, dx=dx[a]) for a in CAND]}
    K = IN["keep166"]["inloop"]
    inl = {"alt": "In-loop Paloma 12-subset clean macro at steps 390279, 390319 and 390350.", "h": 230, "ymin": 1.7460, "ymax": 1.7500, "yticks": [1.746, 1.747, 1.748, 1.749, 1.75], "digits": 4, "ytick": {"digits": 3}, "xmin": 390271, "xmax": 390358, "xticks": [390279, 390319, 390350],
           "xbands": [{"x": int(s), "w": 44, "y0": v - JIT_HI, "y1": v + JIT_HI, "fill": "var(--bandlo)"} for s, v in K.items()] + [{"x": int(s), "w": 44, "y0": v - JIT_LO, "y1": v + JIT_LO, "fill": "var(--bandhi)"} for s, v in K.items()],
           "series": [_ser(a, [[int(s), IN[a]["inloop"][s]] for s in sorted(IN[a]["inloop"], key=int)], dots=True, r=4, thin=True) for a in CAND]}
    return off, inl


def traj_mrcr(RES):
    dx = {"keep166": -9, "global188": -3, "global200": 3, "log32768": 9}
    def spec(alt, key, **kw):
        return {"alt": alt, "h": 220, "xmin": 390241, "xmax": 390361, "xticks": [390251, 390351],
                "series": [_ser(a, [[390251, RES["arms"][a]["base"][key]], [390351, RES["arms"][a]["eval"][key]]], line=False, dots=True, dx=dx[a]) for a in CAND], **kw}
    rec = spec("Recovery at steps 390251 and 390351, not joined.", "rec", ymin=0.2, ymax=1.0, yticks=[0.2, 0.4, 0.6, 0.8, 1.0], digits=3, ytick={"digits": 1})
    ful = spec("Full NLL at steps 390251 and 390351 on a log scale, not joined.", "full_context", ylog=True, ymin=0.02, ymax=1.0, yticks=[0.02, 0.05, 0.1, 0.2, 0.5, 1], ylabels={"0.02": "0.02", "0.05": "0.05", "0.1": "0.1", "0.2": "0.2", "0.5": "0.5", "1": "1"}, digits=4)
    return rec, ful


def block(A, RES, HARD, GRAD, IN, table):
    H = HARD["groups"]
    tab = summary_table(A, RES, HARD, GRAD, table)
    cand = """<h3>What would need to be true for each candidate</h3>
<ul><li><strong>Keep 1.66:</strong> the other candidates' lower full NLL on hard cases would have to be noise or too small to matter. They are 0.0004 to 0.0013 lower at the farthest band and 0.0007 to 0.0020 at 8 needles, with every interval including zero.</li>
<li><strong>Global 1.88:</strong> a full NLL change of -0.0003 [-0.0011, +0.0006] overall would have to be enough, and the 0.0015 [0.0003, 0.0028] higher full NLL on 2 needles at the farthest band would have to be acceptable. Early grad norm reaches 6.31.</li>
<li><strong>Global 2.00:</strong> its hard-case changes would have to hold on another seed: recovery +0.0023 at the farthest band and +0.0031 at 8 needles, full NLL -0.0011 overall. A grad norm of 11.30 on the first step would have to stay harmless over a longer run.</li>
<li><strong>Log 32768:</strong> full NLL of -0.0009 [-0.0019, +0.0000] overall and -0.0020 at 8 needles would have to hold on another seed, and serving support would have to land before export. Early grad norm reaches 5.37.</li></ul>"""
    sep = """<h3>What is and isn't separated</h3>
<ul><li><strong>Merged vs raw:</strong> clean macro is 0.0017 lower on merged at each length, against about 0.003 before training, and raw is higher on all 23 subsets at 262K. Recovery and full NLL do not separate them, overall or in the hard cells.</li>
<li><strong>Clean macro:</strong> does not separate merged recipes. Every arm is within 0.0004 of Keep 1.66 at each length, inside the jitter band.</li>
<li><strong>Overall recovery:</strong> saturated at 0.982 to 0.984. Per-arm intervals overlap, though two paired comparisons exclude zero, and log arms aren't like-for-like (see below).</li>
<li><strong>Hard cases:</strong> some paired changes exclude zero, mainly Global 2.00 on recovery. All are small, uncorrected, and from one seed.</li>
<li><strong>Train CE:</strong> Global 1.88, Global 2.00 and Log 32768 stay below Keep 1.66 by 3.0e-4 to 4.4e-4 over the last 20 steps, beyond the 1.5e-4 one-rack band. The gaps are shrinking, so 100 steps do not show where they settle.</li>
<li><strong>All NLL and recovery intervals:</strong> they resample MRCR sources only. No arm has a repeat, so training noise is not in them.</li></ul>"""
    clean = '<h3>Clean macro against Keep 1.66 merged</h3>' + clean_panels(RES) + '<p class="note">Change from Keep 1.66 merged at step 390,351. Bands around Keep: inner 0.0002, outer 0.0007, the checkpoint jitter on clean macro. Tooltips give each value.</p>'
    full = '<h3>Full NLL at 262K</h3><div class="kpanels">' + nll_panel(RES, "full_context", "Full NLL change from Keep 1.66 merged") + '</div><p class="note">Arm minus Keep 1.66 merged. Dot: paired change. Line: 95% bootstrap interval over 299 MRCR sources. Left of zero is lower NLL.</p>'
    hard = ('<h3>Retrieval on the hard cases</h3><p class="note">Change from Keep 1.66 merged, computed from <code>mrcr_example_losses.jsonl</code>, resampling sources within each cell. Panels share a scale within each metric. The farthest band is evidence over 131,072 tokens away.</p>'
            + hard_panels(HARD) + hard_table(HARD, table) +
            """<ul><li><strong>Farthest band (92 sources):</strong> full NLL does not separate any candidate from Keep 1.66. Recovery does for Global 2.00, +0.0023 [+0.0004, +0.0046].</li>
<li><strong>8 needles (98 sources):</strong> full NLL does not separate any candidate from Keep 1.66. Recovery does for Global 2.00, +0.0031 [+0.0006, +0.0057]. Between candidates, Global 2.00 minus Global 1.88 is -0.0012 [-0.0019, -0.0005] on full NLL.</li>
<li><strong>8 needles at the farthest band (29 sources):</strong> thin. Full NLL intervals exclude zero for Global 2.00 (-0.0042) and Log 32768 (-0.0044), and recovery for Global 2.00 (+0.0083) and Global 1.88 (+0.0052). Dropping any one source keeps the sign, but moves the two full NLL changes to as little as -0.0022 and -0.0020.</li>
<li><strong>The direction depends on needle count:</strong> at 2 needles, raised qk goes the other way. Recovery is lower by 0.0015 to 0.0022 for the three candidates, and full NLL on Global 1.88 is 0.0011 higher.</li>
<li><strong>Multiple comparisons:</strong> across all 19 groups, 28 of 114 candidate-vs-Keep comparisons exclude zero, in both directions. Groups overlap and nothing is corrected.</li>
<li><strong>Too few to read:</strong> the single cells with 13 to 20 sources. They appear only in the full listing under Results.</li>
<li><strong>Log recovery isn't like-for-like:</strong> log arms run their full and distractor rows at higher qk than their needle and query rows. Full NLL has no such issue.</li></ul>""")
    grad = ('<h3>Grad norm, first 20 steps</h3><div class="kpanels">' + _lc("Grad norm per step, log scale", grad_chart(GRAD, 20, 240)) + '</div>'
            '<p class="note">Logged before clipping. The clip threshold is 1.0, so a step at 19.65 is scaled down to norm 1.0 before the optimizer. The table gives steps clipped per arm. The spikes at 390252 land on the same batch for each raised-qk arm; Keep 1.66 reads 0.50 there. Log 65536 arms reach 19.65 and 17.79, not plotted here.</p>')
    return "<h2>Key results</h2>\n<p class=\"note\">All from the merged start except the last row. Candidates first, then the other two log arms and Keep 1.66 raw for reference. Changes are arm minus Keep 1.66 merged. Numbers only; the choice is not made here.</p>\n" + tab + cand + sep + clean + full + hard + grad


def trajectories(RES, GRAD, IN):
    raw, diff, zoom = traj_ce(RES)
    off, inl = traj_eval(RES, IN)
    rec, ful = traj_mrcr(RES)
    return ('<h2>Trajectories over the 100 steps</h2>'
            '<p class="note">One color per candidate, as in Key results. All arms see the same batches.</p>'
            '<h3>Train CE</h3><div class="kpanels">' + _lc("Train CE per step", raw) + '</div>'
            '<p class="note">The four lines nearly coincide: batch-to-batch variation dominates, so read the difference below.</p>'
            '<div class="kpanels">' + _lc("Train CE minus Keep 1.66 merged, all 100 steps", diff) + _lc("Same difference, steps 390271 to 390350, zoomed", zoom) + '</div>'
            '<p class="note">Bands: 1.2e-4 and 1.5e-4, same-queue repeat rms measured on one rack. There is no 11-rack repeat, so this is not measured at this scale. At 11 racks, step-to-step scatter in these differences over the last 50 steps is 5.6e-5 to 8.0e-5, an upper bound on fast noise only. It says nothing about slow drift. Mean over the last 20 steps: Global 1.88 -4.4e-4, Global 2.00 -4.0e-4, Log 32768 -3.0e-4. The first step differs because qk differs on the same weights and batch.</p>'
            '<h3>Grad norm</h3><div class="kpanels">' + _lc("Grad norm per step, log scale", grad_chart(GRAD, 100, 240)) + '</div>'
            '<p class="note">No band: there is no repeat measurement of grad norm at any scale here.</p>'
            '<h3>Held-out loss, from two kinds of eval</h3><div class="kpanels">' + _lc("Offline eval: clean macro, 19 subsets, 262K", off) + _lc("In-loop eval: clean Paloma, 12 subsets", inl) + '</div>'
            '<p class="note">Kept apart on purpose. The in-loop eval runs Paloma only and has no Uncheatable, so it gives 12 of the clean 19 subsets. It also drops each subset\'s final partial window, where offline pads it: <code>ptb</code> and <code>wikitext_103</code> have no in-loop window at all. On the same final weights, in-loop reads 0.0040 above offline for the same 12 subsets, in all seven arms (0.00401 to 0.00405). So differences between arms carry over, and levels do not. Offline points are not joined: nothing was measured between them. Bands sit around Keep 1.66 at each step: inner 0.0002, outer 0.0007, measured offline on the clean 19, and not measured for the in-loop subset.</p>'
            '<h3>MRCR: two measured points only</h3><div class="kpanels">' + _lc("Recovery", rec) + _lc("Full NLL, log scale", ful) + '</div>'
            '<p class="note">Step 390,251 is each arm\'s no-training baseline and step 390,351 its final eval, not joined. No MRCR eval ran in between, and nothing is interpolated. Keep 1.66 starts at recovery 0.301 and full NLL 0.786; the path between is not measured.</p>')


def distractor(RES):
    return '<div class="kpanels">' + nll_panel(RES, "distractor_only", "Distractor NLL change from Keep 1.66 merged (diagnostic)") + "</div>"

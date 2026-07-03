"""
从 time_analysis_data.json 生成 HTML 可视化报告
"""
import json

with open('time_analysis_data.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

html = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>加密货币交易时间窗口实证分析</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;color:#c9d1d9;font-family:-apple-system,"Microsoft YaHei",sans-serif;padding:20px}
.container{max-width:1500px;margin:0 auto}
h1{color:#ff7b72;text-align:center;margin-bottom:6px;font-size:28px}
.subtitle{text-align:center;color:#8b949e;margin-bottom:30px;font-size:13px}
.section{background:#161b22;border-radius:12px;padding:25px;margin-bottom:28px;border:1px solid #30363d}
.section h2{color:#d2a8ff;margin-bottom:18px;font-size:19px;border-bottom:1px solid #30363d;padding-bottom:8px}
.chart-row{display:flex;gap:20px;flex-wrap:wrap}
.chart-box{flex:1;min-width:420px;background:#0d1117;border-radius:8px;padding:15px;border:1px solid #30363d}
.chart-box h3{color:#8b949e;font-size:13px;margin-bottom:10px;text-align:center}
.chart-box canvas{max-height:340px}
table{width:100%;border-collapse:collapse;margin-top:15px;font-size:13px}
th{background:#21262d;color:#d2a8ff;padding:9px 10px;text-align:left;font-weight:600}
td{padding:7px 10px;border-bottom:1px solid #21262d}
tr:hover{background:rgba(210,168,255,0.03)}
.high{color:#ff7b72;font-weight:700}
.green{color:#7ee787}
.amber{color:#d2991d}
.low{color:#8b949e}
.conclusion-box{background:#21262d;border-left:4px solid #d2991d;padding:14px 18px;border-radius:0 8px 8px 0;margin:8px 0;line-height:1.7}
.conclusion-box.good{border-left-color:#7ee787}
.conclusion-box.bad{border-left-color:#ff7b72}
.note{color:#8b949e;font-size:12px;margin-top:6px}
.finding-row{display:flex;gap:15px;flex-wrap:wrap;margin-bottom:15px}
.finding-tag{display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600}
.tag-yes{background:rgba(126,231,135,0.15);color:#7ee787}
.tag-no{background:rgba(255,123,114,0.15);color:#ff7b72}
.tag-maybe{background:rgba(210,169,29,0.15);color:#d2991d}
</style>
</head>
<body>
<div class="container">
<h1>加密货币交易时间窗口实证分析</h1>
<p class="subtitle">数据来源: Gate交易所 1分钟K线 | 北京时间(UTC+8) | 2026年6月 | 18个币种, 85,214条工作日数据</p>

<div class="section">
<h2>核心结论速览</h2>
<div class="finding-row">
<span class="finding-tag tag-yes">✓ 美盘高波动:支持</span>
<span class="finding-tag tag-yes">✓ 整点效应:支持</span>
<span class="finding-tag tag-yes">✓ 午餐低波动:部分支持</span>
<span class="finding-tag tag-maybe">△ 亚盘活跃:弱</span>
<span class="finding-tag tag-no">✗ 欧盘最佳:不支持</span>
<span class="finding-tag tag-no">✗ 凌晨稳定:不支持</span>
<span class="finding-tag tag-no">✗ 周一延后:不支持</span>
</div>
<p class="note">基于18个币种在Gate交易所的实际1分钟K线数据的统计分析。所有时间为北京时间(UTC+8)。</p>
</div>

<div class="section">
<h2>一、24小时波动率分布 (工作日)</h2>
<div class="chart-row">
<div class="chart-box"><h3>三大类别波动率 x 小时</h3><canvas id="cHourlyRange"></canvas></div>
<div class="chart-box"><h3>三大类别成交量 x 小时</h3><canvas id="cHourlyVol"></canvas></div>
</div>
<div class="chart-row" style="margin-top:20px">
<div class="chart-box"><h3>长影线K线占比 x 小时</h3><canvas id="cHourlyWick"></canvas></div>
</div>
</div>

<div class="section">
<h2>二、博主声称时段 vs 实证数据</h2>
<div class="conclusion-box good">
<strong>验证方法:</strong> 计算各时段波动率与全天均值的比值。&gt;1.10 = 高波动(支持做单时段), &lt;0.90 = 低波动(支持垃圾时段), 0.90~1.10 = 接近均值(无明显特征)。
</div>
<table id="zoneTable"><thead><tr><th>时段</th><th>类别</th><th>范围均值%</th><th>vs全天均值</th><th>长影线%</th><th>成交量</th><th>验证</th></tr></thead><tbody id="zoneBody"></tbody></table>
<p class="note">比值 = 该时段波动率 / 全天均值</p>
</div>

<div class="section">
<h2>三、整点效应<span style="font-size:13px;color:#8b949e;margin-left:10px">(博主: 整点前5-10分钟谨慎开单)</span></h2>
<div class="chart-row">
<div class="chart-box"><h3>整点±5min vs 远离整点 — 波动率对比</h3><canvas id="cRoundRange"></canvas></div>
<div class="chart-box"><h3>整点±5min vs 远离整点 — 长影线占比</h3><canvas id="cRoundWick"></canvas></div>
</div>
<div id="roundTableBox"></div>
</div>

<div class="section">
<h2>四、周一亚盘延迟效应<span style="font-size:13px;color:#8b949e;margin-left:10px">(博主: 周一8:30延迟到9:30)</span></h2>
<div class="chart-row">
<div class="chart-box"><h3>周一 vs 周二~五: 8:30-9:30 波动率对比</h3><canvas id="cMondayRange"></canvas></div>
<div class="chart-box"><h3>周一 vs 周二~五: 成交量对比</h3><canvas id="cMondayVol"></canvas></div>
</div>
<div id="mondayTableBox"></div>
</div>

<div class="section">
<h2>五、凌晨稳定性<span style="font-size:13px;color:#8b949e;margin-left:10px">(博主: 0:00-1:00适合新人)</span></h2>
<div class="chart-row">
<div class="chart-box"><h3>凌晨 vs 白天: 波动率标准差(越小越稳定)</h3><canvas id="cMidnightStd"></canvas></div>
<div class="chart-box"><h3>凌晨 vs 白天: 长影线</h3><canvas id="cMidnightWick"></canvas></div>
</div>
<div id="midnightTableBox"></div>
</div>

<div class="section">
<h2>六、美盘风险特征<span style="font-size:13px;color:#8b949e;margin-left:10px">(博主: 对新人很不友好)</span></h2>
<table id="usTable"><thead><tr><th>类别</th><th>美盘均值%</th><th>美盘极限%</th><th>美盘大波动%</th><th>非美盘均值%</th><th>非美盘大波动%</th><th>风险倍数</th></tr></thead><tbody id="usBody"></tbody></table>
</div>

<div class="section">
<h2>七、各币种平均波动率排名</h2>
<table><thead><tr><th>排名</th><th>币种</th><th>类别</th><th>平均波动率%</th><th>定位</th></tr></thead><tbody id="coinBody"></tbody></table>
</div>

<div class="section">
<h2>八、一周各天波动率</h2>
<table id="dowTable"><thead><tr><th>类别</th><th>周一</th><th>周二</th><th>周三</th><th>周四</th><th>周五</th></tr></thead><tbody id="dowBody"></tbody></table>
</div>

<div class="section">
<h2>九、总体结论与建议</h2>
<div class="conclusion-box good">
<strong>结论1: 美盘(21:30-23:30)是所有币种中波动率最高的时段 ✓</strong><br>
博主此说法完全成立。稳定型币在美盘波动率放大到全天均值的2.1倍，中间币1.6倍，妖币1.1倍。但对新人不友好的原因是波动大、方向变化快，而非插针。<br>
<span style="color:#7ee787">建议: 美盘做趋势交易,放大止损;新手降低杠杆。</span>
</div>

<div class="conclusion-box good">
<strong>结论2: 整点前后5分钟波动率确实放大 (3.6%~6.5%) ✓</strong><br>
博主的"整点前5-10分钟谨慎开单"建议在所有类别中都成立。整点±5min的波动率和长影线占比均高于远离整点。<br>
<span style="color:#7ee787">建议: 策略中可加入整点过滤器,推迟开仓至整点后3-5分钟。</span>
</div>

<div class="conclusion-box">
<strong>结论3: 午餐时段(11:00-13:00)对稳定型和中间币确实是低波动期 △</strong><br>
稳定型币波动降至全天均值的84%,中间币降至78%,符合博主"垃圾时间"描述。但对妖币不成立(91%)。<br>
<span style="color:#d2991d">建议: 对主流币可此时降低仓位,但妖币不受此规律约束。</span>
</div>

<div class="conclusion-box bad">
<strong>结论4: 凌晨0:00-1:00非但不稳定,反而波动率偏大 ✗</strong><br>
博主推荐新人此时交易,但实证数据显示:稳定型凌晨波动标准差比白天高18%,妖币高16%。只有中间币略低(82%)。<br>
"凌晨稳定"这个说法在加密货币市场不成立,甚至相反。<br>
<span style="color:#ff7b72">建议: 不要相信凌晨稳定的说法!稳定型币在0-1时的波动率反而是全天第五高。</span>
</div>

<div class="conclusion-box bad">
<strong>结论5: 欧盘15:30-16:30不是加密货币的最佳交易时段 ✗</strong><br>
博主将此称为最佳做单时间,但实证显示:稳定型币此时波动率仅为全天均值的71%,是全天最低的时段之一。<br>
中间币此时略高(1.12),妖币接近均值(0.98)。整体而言,这并非加密货币的交易黄金时段。<br>
<span style="color:#ff7b72">建议: 加密货币的最佳时段是美盘21:30-23:30,而非欧盘15:30-16:30。</span>
</div>

<div class="conclusion-box bad">
<strong>结论6: 周一亚盘延迟1小时的说法不成立 ✗</strong><br>
博主认为周一8:30-9:30波动不足,应延迟到9:30。但数据显示:稳定型周一8:30-9:30波动率反而是平时的154%!中间币112%。<br>
周一不仅不需要延后,反而波动比平时更大(可能是消化周末消息所致)。<br>
<span style="color:#ff7b72">建议: 周一开盘第一时间反而是波动窗口,不应避开。</span>
</div>

<div class="conclusion-box">
<strong>结论7: 亚盘8:30-10:30在加密货币中并非显著高波动期 △</strong><br>
三类币种的波动率均接近或略低于全天均值(稳定型0.92,中间币0.85,妖币0.94)。博主的亚盘活跃时段对黄金可能适用,但对加密货币无明显优势。
</div>

<div class="conclusion-box good" style="margin-top:20px;border-left-color:#7ee787">
<strong>综合建议: 适合加密货币的各时段策略</strong><br><br>
<table style="margin-top:0">
<tr><th>时段(北京时间)</th><th>特征</th><th>适合策略</th><th>风险</th></tr>
<tr><td>07:00-10:00</td><td>亚盘消化隔夜消息,周一波动特别大</td><td>趋势跟进</td><td>假突破</td></tr>
<tr><td>11:00-13:00</td><td>低波动(稳定型/中间币)</td><td>网格/区间</td><td>流动性不足</td></tr>
<tr><td>15:00-17:00</td><td>欧亚重叠,波动不定</td><td>谨慎观望</td><td>方向不明</td></tr>
<tr><td style="color:#7ee787">21:30-23:30</td><td style="color:#7ee787">最高波动时段</td><td style="color:#7ee787">趋势/突破策略</td><td>剧烈反转</td></tr>
<tr><td>00:00-01:00</td><td>波动不低!误判区</td><td>不建议新手</td><td>深夜流动性差</td></tr>
</table>
</div>

</div>
</div>

<script>
const D = ''' + json.dumps(d, ensure_ascii=False) + ''';

// Hourly charts
const colors = {稳定型:'#7ee787',中间币:'#d2991d',妖币:'#ff7b72'};
const alphas = {稳定型:'rgba(126,231,135,0.08)',中间币:'rgba(210,153,29,0.08)',妖币:'rgba(255,123,114,0.08)'};

function makeLine(canvasId, key, title) {
    const datasets = ['稳定型','中间币','妖币'].map(c=>({
        label:c, data:D.hourly[c][key], borderColor:colors[c],
        backgroundColor:alphas[c], tension:0.3, fill:true, pointRadius:0
    }));
    new Chart(document.getElementById(canvasId), {
        type:'line', data:{labels:D.hourly.稳定型.hours, datasets},
        options:{
            responsive:true,
            plugins:{legend:{labels:{color:'#8b949e'}}},
            scales:{
                x:{title:{display:true,text:'北京时间(小时)',color:'#8b949e'},ticks:{color:'#8b949e',maxTicksLimit:24}},
                y:{title:{display:true,text:title,color:'#8b949e'},ticks:{color:'#8b949e'}}
            }
        }
    });
}
makeLine('cHourlyRange','range','平均波动率(%)');
makeLine('cHourlyVol','volume','平均成交量');
// Wick as bar chart
new Chart(document.getElementById('cHourlyWick'), {
    type:'bar',
    data:{labels:D.hourly.稳定型.hours, datasets:['稳定型','中间币','妖币'].map(c=>({label:c,data:D.hourly[c].wick,backgroundColor:colors[c]}))},
    options:{
        responsive:true,
        plugins:{legend:{labels:{color:'#8b949e'}}},
        scales:{
            x:{title:{display:true,text:'北京时间(小时)',color:'#8b949e'},ticks:{color:'#8b949e',maxTicksLimit:24}},
            y:{title:{display:true,text:'长影线占比(%)',color:'#8b949e'},ticks:{color:'#8b949e'}}
        }
    }
});

// Zone table
let zoneHTML='', lastZ='';
for(const z of D.zones){
    let v='';
    if(z.ratio>1.10) v='<span class="high">高波动</span>';
    else if(z.ratio<0.90) v='<span class="green">低波动</span>';
    else v='<span class="amber">≈均值</span>';
    const zl = z.zone===lastZ?'':z.zone;
    lastZ=z.zone;
    zoneHTML+=`<tr><td>${zl}</td><td>${z.cat}</td><td>${z.range}%</td><td>${(z.ratio*100).toFixed(1)}%</td><td>${z.wick}%</td><td>${z.vol}</td><td>${v}</td></tr>`;
}
document.getElementById('zoneBody').innerHTML=zoneHTML;

// Round hour charts
const rc = D.round.map(d=>d.cat);
new Chart(document.getElementById('cRoundRange'), {
    type:'bar',
    data:{labels:rc, datasets:[
        {label:'整点±5min',data:D.round.map(d=>d.near_range),backgroundColor:'#ff7b72'},
        {label:'远离整点',data:D.round.map(d=>d.far_range),backgroundColor:'#7ee787'},
    ]},
    options:{responsive:true,plugins:{legend:{labels:{color:'#8b949e'}}},scales:{y:{ticks:{color:'#8b949e'}}}}
});
new Chart(document.getElementById('cRoundWick'), {
    type:'bar',
    data:{labels:rc, datasets:[
        {label:'整点±5min',data:D.round.map(d=>d.near_wick),backgroundColor:'#ff7b72'},
        {label:'远离整点',data:D.round.map(d=>d.far_wick),backgroundColor:'#7ee787'},
    ]},
    options:{responsive:true,plugins:{legend:{labels:{color:'#8b949e'}}},scales:{y:{ticks:{color:'#8b949e'}}}}
});
let rtHTML='<table><thead><tr><th>类别</th><th>整点±5min范围%</th><th>远离整点%</th><th>差异</th><th>整点长影线%</th><th>远离长影线%</th></tr></thead><tbody>';
for(const r of D.round){
    const delta=((r.near_range/r.far_range-1)*100).toFixed(1);
    rtHTML+=`<tr><td>${r.cat}</td><td>${r.near_range}%</td><td>${r.far_range}%</td><td><span class="${delta>0?'high':'green'}">${delta>0?'+':''}${delta}%</span></td><td>${r.near_wick}%</td><td>${r.far_wick}%</td></tr>`;
}
rtHTML+='</tbody></table>';
document.getElementById('roundTableBox').innerHTML=rtHTML;

// Monday charts
new Chart(document.getElementById('cMondayRange'), {
    type:'bar',
    data:{labels:D.monday.map(d=>d.cat), datasets:[
        {label:'周一 8:30-9:30',data:D.monday.map(d=>d.mon_early_range),backgroundColor:'#ff7b72'},
        {label:'周一 9:30-10:30',data:D.monday.map(d=>d.mon_late_range),backgroundColor:'#d2991d'},
        {label:'周二~五 8:30-9:30',data:D.monday.map(d=>d.tf_early_range),backgroundColor:'#7ee787'},
    ]},
    options:{responsive:true,plugins:{legend:{labels:{color:'#8b949e'}}},scales:{y:{ticks:{color:'#8b949e'}}}}
});
new Chart(document.getElementById('cMondayVol'), {
    type:'bar',
    data:{labels:D.monday.map(d=>d.cat), datasets:[
        {label:'周一 8:30-9:30',data:D.monday.map(d=>d.mon_early_vol),backgroundColor:'#ff7b72'},
        {label:'周一 9:30-10:30',data:D.monday.map(d=>d.mon_late_vol),backgroundColor:'#d2991d'},
        {label:'周二~五 8:30-9:30',data:D.monday.map(d=>d.tf_early_vol),backgroundColor:'#7ee787'},
    ]},
    options:{responsive:true,plugins:{legend:{labels:{color:'#8b949e'}}},scales:{y:{ticks:{color:'#8b949e'}}}}
});
let monHTML='<table><thead><tr><th>类别</th><th>周一8:30范围%</th><th>周一9:30范围%</th><th>周二~五8:30范围%</th><th>周一/平时比</th><th>结论</th></tr></thead><tbody>';
for(const m of D.monday){
    const ratio=m.mon_early_range/m.tf_early_range;
    const rpct=(ratio*100).toFixed(0);
    let v='';
    if(ratio<0.85) v='<span class="green">周一确实更低(支持博主的延迟建议)</span>';
    else if(ratio>1.15) v='<span class="high">周一反而更高!不支持延迟</span>';
    else v='<span class="amber">差异不显著</span>';
    monHTML+=`<tr><td>${m.cat}</td><td>${m.mon_early_range}%</td><td>${m.mon_late_range}%</td><td>${m.tf_early_range}%</td><td>${rpct}%</td><td>${v}</td></tr>`;
}
monHTML+='</tbody></table>';
document.getElementById('mondayTableBox').innerHTML=monHTML;

// Midnight charts
new Chart(document.getElementById('cMidnightStd'), {
    type:'bar',
    data:{labels:D.midnight.map(d=>d.cat), datasets:[
        {label:'凌晨0-1时 Std',data:D.midnight.map(d=>d.mid_std),backgroundColor:'#d2991d'},
        {label:'白天1-24时 Std',data:D.midnight.map(d=>d.day_std),backgroundColor:'#7ee787'},
    ]},
    options:{responsive:true,plugins:{legend:{labels:{color:'#8b949e'}}},scales:{y:{ticks:{color:'#8b949e'}}}}
});
new Chart(document.getElementById('cMidnightWick'), {
    type:'bar',
    data:{labels:D.midnight.map(d=>d.cat), datasets:[
        {label:'凌晨0-1时',data:D.midnight.map(d=>d.mid_wick),backgroundColor:'#d2991d'},
        {label:'白天',data:D.midnight.map(d=>d.day_wick),backgroundColor:'#7ee787'},
    ]},
    options:{responsive:true,plugins:{legend:{labels:{color:'#8b949e'}}},scales:{y:{ticks:{color:'#8b949e'}}}}
});
let midHTML='<table><thead><tr><th>类别</th><th>凌晨范围%</th><th>凌晨Std</th><th>白天范围%</th><th>白天Std</th><th>凌晨长影线%</th><th>白天长影线%</th><th>稳定性判断</th></tr></thead><tbody>';
for(const m of D.midnight){
    const ratio=m.mid_std/m.day_std;
    let v;
    if(ratio<0.85) v='<span class="green">凌晨更稳定</span>';
    else if(ratio<1.0) v='<span class="amber">凌晨略稳</span>';
    else v='<span class="high">凌晨反而不稳定!</span>';
    midHTML+=`<tr><td>${m.cat}</td><td>${m.mid_range}%</td><td>${m.mid_std}</td><td>${m.day_range}%</td><td>${m.day_std}</td><td>${m.mid_wick}%</td><td>${m.day_wick}%</td><td>${v}</td></tr>`;
}
midHTML+='</tbody></table>';
document.getElementById('midnightTableBox').innerHTML=midHTML;

// US risk table
let usHTML='';
for(const u of D.us){
    const ratio=u.us_large_pct/u.other_large_pct;
    usHTML+=`<tr><td>${u.cat}</td><td>${u.us_range}%</td><td>${u.us_max}%</td><td>${u.us_large_pct}%</td><td>${u.other_range}%</td><td>${u.other_large_pct}%</td><td><span class="high">${ratio.toFixed(1)}x</span></td></tr>`;
}
document.getElementById('usBody').innerHTML=usHTML;

// Coin ranking
const cr=D.coin_rank;
let crHTML='';
cr.forEach((c,i)=>{
    let pos;
    if(i<6) pos='<span class="high">高波动</span>';
    else if(i<12) pos='<span class="amber">中等</span>';
    else pos='<span class="green">低波动</span>';
    crHTML+=`<tr><td>${i+1}</td><td>${c.coin}</td><td>${c.cat}</td><td>${c.range}%</td><td>${pos}</td></tr>`;
});
document.getElementById('coinBody').innerHTML=crHTML;

// Day of week
let dowHTML='';
for(const x of D.dow){
    const vals=['周一','周二','周三','周四','周五'].map((_,i)=>x[`d${i}_range`].toFixed(5)+'%');
    const maxV=Math.max(...vals.map(v=>parseFloat(v)));
    const minV=Math.min(...vals.map(v=>parseFloat(v)));
    dowHTML+=`<tr><td>${x.cat}</td>`;
    ['周一','周二','周三','周四','周五'].forEach((_,i)=>{
        const v=parseFloat(x[`d${i}_range`]);
        let cls='';
        if(v===maxV) cls='high';
        else if(v===minV) cls='green';
        dowHTML+=`<td class="${cls}">${x[`d${i}_range`].toFixed(5)}%</td>`;
    });
    dowHTML+='</tr>';
}
document.getElementById('dowBody').innerHTML=dowHTML;
</script>
</div>
</body>
</html>'''

with open('time_analysis_report.html', 'w', encoding='utf-8') as f:
    f.write(html)
print('报告已生成: time_analysis_report.html')

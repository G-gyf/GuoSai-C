import fs from 'node:fs/promises';
import path from 'node:path';
import { FileBlob, SpreadsheetFile } from '@oai/artifact-tool';

const root = path.resolve(import.meta.dirname, '../../../..');
const outputFlag = process.argv.indexOf('--output-dir');
const out = outputFlag >= 0 ? path.resolve(process.argv[outputFlag + 1]) : import.meta.dirname;
const wb = await SpreadsheetFile.importXlsx(await FileBlob.load(path.join(root,'附件/附件5/result2.xlsx')));
console.log((await wb.inspect({kind:'workbook,sheet,table',maxChars:1500,tableMaxRows:2,tableMaxCols:3})).ndjson);
if (process.argv.includes('--inspect')) process.exit(0);

const p = JSON.parse(await fs.readFile(path.join(out,'question2_workbook_payload.json'),'utf8'));
const plan = wb.worksheets.getItem('计划购电量');
const battery = wb.worksheets.getItem('充放电量');
const emergency = wb.worksheets.getItem('紧急购电量');
await fs.mkdir(path.join(out,'qa'),{recursive:true});
const template = await wb.render({sheetName:'充放电量',range:'A1:F8',scale:1.5,format:'png'});
await fs.writeFile(path.join(out,'qa/template.png'),new Uint8Array(await template.arrayBuffer()));

// Keep the three official worksheets, headers and the purchase column order.
plan.getRange('B2:EO335').values=p.purchases.map(r=>r.slice(1,145));
plan.getRange('EP2:EP335').values=p.purchases.map(r=>[r[145]]);
plan.getRange('EQ2:EQ335').values=p.purchases.map(r=>[r[146]]);
plan.getRange('B2:EQ335').setNumberFormat('0.000000');

function serial(date) { return (Date.parse(date+'T00:00:00Z')-Date.UTC(1899,11,30))/86400000; }
battery.getRange('A2:F20').clear({applyTo:'contents'});
for(let d=0;d<334;d++) {
  const start=2+6*d;
  if(d>0) battery.getRange(`A${start}:F${start+5}`).copyFrom(battery.getRange('A2:F7'),'all');
}
battery.getRange(`A2:F${p.batteries.length+1}`).values=p.batteries.map(r=>[r[0]?serial(r[0]):null,...r.slice(1)]);
battery.getRange(`A2:A${p.batteries.length+1}`).setNumberFormat('yyyy/m/d');
battery.getRange(`C2:D${p.batteries.length+1}`).setNumberFormat('0.000000');
battery.getRange(`F2:F${p.batteries.length+1}`).setNumberFormat('0.000000');
emergency.getRange('A2:C11').clear({applyTo:'contents'});
for(let i=1;i<p.emergency.length;i++) {
  emergency.getRange(`A${i+2}:C${i+2}`).copyFrom(emergency.getRange('A2:C2'),'all');
}
emergency.getRange(`A2:C${p.emergency.length+1}`).values=p.emergency.map(r=>[serial(r[0]),r[1],r[2]]);
emergency.getRange(`A2:A${p.emergency.length+1}`).setNumberFormat('yyyy/m/d');
emergency.getRange(`C2:C${p.emergency.length+1}`).setNumberFormat('0.000000');

// Only extend/fit the affected result areas; no added analytical worksheets.
for(const [s,range] of [[plan,'A1:EQ335'],[battery,`A1:F${p.batteries.length+1}`],[emergency,`A1:C${p.emergency.length+1}`]]) {
  s.getRange(range).format.rowHeight=21;
  s.freezePanes.freezeRows(1);
}
plan.getRange('B1:EQ335').format.columnWidth=19;
plan.getRange('A1:A335').format.columnWidth=15;
battery.getRange(`A1:F${p.batteries.length+1}`).format.columnWidth=19;
emergency.getRange(`A1:C${p.emergency.length+1}`).format.columnWidth=21;
wb.recalculate();
for(const [name,range] of [['计划购电量','A1:F8'],['充放电量','A1:F13'],['紧急购电量','A1:C15'],['计划购电量','EK1:EQ8']]) {
 const img=await wb.render({sheetName:name,range,scale:1.5,format:'png'});
 await fs.writeFile(path.join(out,'qa',`${name}-${range.replace(':','_')}.png`),new Uint8Array(await img.arrayBuffer()));
}
const result=await SpreadsheetFile.exportXlsx(wb);
await result.save(path.join(out,'result2.xlsx'));
console.log(JSON.stringify({output:'result2.xlsx',days:p.purchases.length,batteryRows:p.batteries.length,emergencyRows:p.emergency.length}));
process.exit(0);

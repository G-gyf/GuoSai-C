import fs from 'node:fs/promises';
import path from 'node:path';
import { FileBlob, SpreadsheetFile } from '@oai/artifact-tool';

// Question 3 result3.xlsx backfill.  Run from the project root:
//   node outputs/question3/build_workbook_q3.mjs --output-dir outputs/question3/current
// The four official worksheets are kept: 计划购电量, 调整购电量, 充放电量, 紧急购电量.
const root = path.resolve(import.meta.dirname, '../..');
const outputFlag = process.argv.indexOf('--output-dir');
const out = outputFlag >= 0 ? path.resolve(process.argv[outputFlag + 1]) : import.meta.dirname;
const wb = await SpreadsheetFile.importXlsx(await FileBlob.load(path.join(root, '附件/附件5/result3.xlsx')));
if (process.argv.includes('--inspect')) {
  console.log((await wb.inspect({ kind: 'workbook,sheet,table', maxChars: 1200, tableMaxRows: 2, tableMaxCols: 3 })).ndjson);
  process.exit(0);
}

const p = JSON.parse(await fs.readFile(path.join(out, 'question3_workbook_payload.json'), 'utf8'));
const plan = wb.worksheets.getItem('计划购电量');
const adjusted = wb.worksheets.getItem('调整购电量');
const battery = wb.worksheets.getItem('充放电量');
const emergency = wb.worksheets.getItem('紧急购电量');
await fs.mkdir(path.join(out, 'qa'), { recursive: true });

function serial(date) { return (Date.parse(date + 'T00:00:00Z') - Date.UTC(1899, 11, 30)) / 86400000; }

// 计划购电量: q0 intervals (template order) + 全天购电量 + 全天购电费 (face value)
plan.getRange('B2:EO335').values = p.purchases.map(r => r.slice(1, 145));
plan.getRange('EP2:EP335').values = p.purchases.map(r => [r[145]]);
plan.getRange('EQ2:EQ335').values = p.purchases.map(r => [r[146]]);
plan.getRange('B2:EQ335').setNumberFormat('0.000000');

// 调整购电量: final qA intervals + 全天购电量 + 全天购电费 (segment net settlement)
adjusted.getRange('B2:EO335').values = p.adjusted.map(r => r.slice(1, 145));
adjusted.getRange('EP2:EP335').values = p.adjusted.map(r => [r[145]]);
adjusted.getRange('EQ2:EQ335').values = p.adjusted.map(r => [r[146]]);
adjusted.getRange('B2:EQ335').setNumberFormat('0.000000');

// 充放电量: 334 days x 6 four-hour blocks
battery.getRange('A2:F26').clear({ applyTo: 'contents' });
for (let d = 1; d < 334; d++) {
  const start = 2 + 6 * d;
  battery.getRange(`A${start}:F${start + 5}`).copyFrom(battery.getRange('A2:F7'), 'all');
}
battery.getRange(`A2:F${p.batteries.length + 1}`).values = p.batteries.map(r => [r[0] ? serial(r[0]) : null, ...r.slice(1)]);
battery.getRange(`A2:A${p.batteries.length + 1}`).setNumberFormat('yyyy/m/d');
battery.getRange(`C2:D${p.batteries.length + 1}`).setNumberFormat('0.000000');
battery.getRange(`F2:F${p.batteries.length + 1}`).setNumberFormat('0.000000');

// 紧急购电量: merged intervals plus '无' rows
emergency.getRange('A2:C11').clear({ applyTo: 'contents' });
for (let i = 1; i < p.emergency.length; i++) {
  emergency.getRange(`A${i + 2}:C${i + 2}`).copyFrom(emergency.getRange('A2:C2'), 'all');
}
emergency.getRange(`A2:C${p.emergency.length + 1}`).values = p.emergency.map(r => [serial(r[0]), r[1], r[2]]);
emergency.getRange(`A2:A${p.emergency.length + 1}`).setNumberFormat('yyyy/m/d');
emergency.getRange(`C2:C${p.emergency.length + 1}`).setNumberFormat('0.000000');

for (const [s, range] of [
  [plan, 'A1:EQ335'],
  [adjusted, 'A1:EQ335'],
  [battery, `A1:F${p.batteries.length + 1}`],
  [emergency, `A1:C${p.emergency.length + 1}`],
]) {
  s.getRange(range).format.rowHeight = 21;
  s.freezePanes.freezeRows(1);
}
plan.getRange('B1:EQ335').format.columnWidth = 19;
plan.getRange('A1:A335').format.columnWidth = 15;
adjusted.getRange('B1:EQ335').format.columnWidth = 19;
adjusted.getRange('A1:A335').format.columnWidth = 15;
battery.getRange(`A1:F${p.batteries.length + 1}`).format.columnWidth = 19;
emergency.getRange(`A1:C${p.emergency.length + 1}`).format.columnWidth = 21;
wb.recalculate();
for (const [name, range] of [
  ['计划购电量', 'A1:F8'],
  ['调整购电量', 'A1:F8'],
  ['充放电量', 'A1:F13'],
  ['紧急购电量', 'A1:C15'],
  ['调整购电量', 'EK1:EQ8'],
]) {
  const img = await wb.render({ sheetName: name, range, scale: 1.5, format: 'png' });
  await fs.writeFile(path.join(out, 'qa', `${name}-${range.replace(':', '_')}.png`), new Uint8Array(await img.arrayBuffer()));
}
const result = await SpreadsheetFile.exportXlsx(wb);
await result.save(path.join(out, 'result3.xlsx'));
console.log(JSON.stringify({ output: 'result3.xlsx', days: p.purchases.length, batteryRows: p.batteries.length, emergencyRows: p.emergency.length }));
process.exit(0);

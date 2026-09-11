import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";


function parseArgs(argv) {
  const result = {};
  for (let i = 2; i < argv.length; i += 2) {
    const key = argv[i];
    const value = argv[i + 1];
    if (!key?.startsWith("--") || value === undefined) {
      throw new Error(`Invalid argument near ${key ?? "end of command"}`);
    }
    result[key.slice(2)] = value;
  }
  return result;
}


const args = parseArgs(process.argv);
for (const required of ["template", "preview-dir"]) {
  if (!args[required]) throw new Error(`Missing --${required}`);
}

const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(args.template));
const sheetNames = workbook.worksheets.items.map((sheet) => sheet.name);
if (sheetNames.length !== 2 || !sheetNames.includes("计划购电量") || !sheetNames.includes("充放电量")) {
  throw new Error(`Unexpected result1 template sheets: ${sheetNames.join(", ")}`);
}

if (args["inspect-only"] === "true") {
  const inspection = await workbook.inspect({
    kind: "workbook,sheet,table,formula,computedStyle",
    maxChars: 10000,
    tableMaxRows: 10,
    tableMaxCols: 8,
  });
  await fs.mkdir(args["preview-dir"], { recursive: true });
  for (const sheetName of sheetNames) {
    const preview = await workbook.render({
      sheetName,
      autoCrop: "all",
      scale: 1.5,
      format: "png",
    });
    await fs.writeFile(
      path.join(args["preview-dir"], `template-${sheetName}.png`),
      new Uint8Array(await preview.arrayBuffer()),
    );
  }
  console.log(inspection.ndjson);
  process.exit(0);
}

for (const required of ["data", "output"]) {
  if (!args[required]) throw new Error(`Missing --${required}`);
}
const payload = JSON.parse(await fs.readFile(args.data, "utf8"));
if (!Array.isArray(payload.purchase_kwh) || payload.purchase_kwh.length !== 144) {
  throw new Error("purchase_kwh must contain exactly 144 values");
}
if (!Array.isArray(payload.blocks) || payload.blocks.length !== 6) {
  throw new Error("blocks must contain exactly 6 four-hour summaries");
}

const purchaseSheet = workbook.worksheets.getItem("计划购电量");
const storageSheet = workbook.worksheets.getItem("充放电量");
purchaseSheet.getRange("B2:B145").values = payload.purchase_kwh.map((value) => [Number(value)]);
storageSheet.getRange("B2:C7").values = payload.blocks.map((row) => [
  Number(row.charge_kwh),
  Number(row.discharge_kwh),
]);
storageSheet.getRange("E2:E3").values = [
  [Number(payload.soc_0_kwh)],
  [Number(payload.soc_24_kwh)],
];

purchaseSheet.getRange("B2:B145").setNumberFormat("0.000000");
storageSheet.getRange("B2:C7").setNumberFormat("0.000000");
storageSheet.getRange("E2:E3").setNumberFormat("0.00");

workbook.recalculate();

const keyRanges = await workbook.inspect({
  kind: "table",
  range: "计划购电量!A1:B145",
  include: "values,formulas",
  tableMaxRows: 8,
  tableMaxCols: 2,
  maxChars: 5000,
});
const storageRanges = await workbook.inspect({
  kind: "table",
  range: "充放电量!A1:E7",
  include: "values,formulas",
  tableMaxRows: 10,
  tableMaxCols: 6,
  maxChars: 5000,
});
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 100 },
  summary: "Question 1 final formula error scan",
});

await fs.mkdir(args["preview-dir"], { recursive: true });
for (const sheetName of sheetNames) {
  const preview = await workbook.render({
    sheetName,
    autoCrop: "all",
    scale: 1.5,
    format: "png",
  });
  await fs.writeFile(
    path.join(args["preview-dir"], `${sheetName}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

await fs.mkdir(path.dirname(args.output), { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(args.output);

const purchaseValues = purchaseSheet.getRange("B2:B145").values;
const storageValues = storageSheet.getRange("B2:E7").values;
const numericPurchases = purchaseValues.flat().filter((value) => typeof value === "number");
if (numericPurchases.length !== 144) throw new Error("Workbook lost purchase values before export");

console.log(JSON.stringify({
  output: args.output,
  sheets: sheetNames,
  purchase_value_count: numericPurchases.length,
  purchase_total_kwh: numericPurchases.reduce((sum, value) => sum + value, 0),
  storage_values: storageValues,
  purchase_inspect: keyRanges.ndjson,
  storage_inspect: storageRanges.ndjson,
  formula_error_scan: errors.ndjson,
}, null, 2));
process.exit(0);

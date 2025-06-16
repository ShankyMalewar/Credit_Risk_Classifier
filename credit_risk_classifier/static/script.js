const features = [
  { key: "num__AMT_CREDIT_SUM_sum", label: "Total credit you’ve taken so far (₹)" },
  { key: "num__AMT_CREDIT_SUM_mean", label: "Average credit amount per account (₹)" },
  { key: "num__INST_AMT_PAYMENT_MAX", label: "Largest installment you paid (₹)" },
  { key: "num__INST_AMT_PAYMENT_MEAN", label: "Average installment you paid (₹)" },
  { key: "num__AMT_CREDIT_SUM_DEBT_max", label: "Maximum debt on any credit (₹)" },
  { key: "num__POS_MONTHS_BALANCE_MEAN", label: "Average active credit duration (months)" },
  { key: "num__INST_PAYMENT_RATIO_MIN", label: "Minimum payment-to-installment ratio" },
  { key: "num__SK_ID_BUREAU_sum", label: "Number of bureau reports" },
  { key: "num__INST_AMT_INSTALMENT_MAX", label: "Largest planned installment (₹)" },
  { key: "num__INST_AMT_INSTALMENT_SUM", label: "Sum of all planned installments (₹)" },
  { key: "num__AMT_CREDIT_SUM_DEBT_mean", label: "Average debt across credits (₹)" },
  { key: "num__POS_MONTHS_BALANCE_MIN", label: "Shortest active credit duration (months)" },
  { key: "num__AMT_CREDIT_SUM_DEBT_sum", label: "Sum of all debts (₹)" },
  { key: "cat__NAME_EDUCATION_TYPE_Higher education", label: "Do you have higher education?", type: "select", options: [{ val: 1, label: "Yes" }, { val: 0, label: "No" }] },
  { key: "num__INST_AMT_PAYMENT_SUM", label: "Total paid on installments (₹)" },
  { key: "num__POS_MONTHS_BALANCE_SUM", label: "Total active credit months" },
  { key: "num__DAYS_CREDIT_sum", label: "Total days since all credits" },
  { key: "num__DAYS_CREDIT_min", label: "Minimum days since credit" },
  { key: "num__DAYS_CREDIT_max", label: "Maximum days since credit" },
  { key: "num__DAYS_CREDIT_mean", label: "Average days since credit" },
];

const form = document.getElementById("credit-form");
const feedback = document.getElementById("feedback");

features.forEach(field => {
  const label = document.createElement("label");
  label.innerText = field.label;

  let input;
  if (field.type === "select") {
    input = document.createElement("select");
    field.options.forEach(opt => {
      const option = document.createElement("option");
      option.value = opt.val;
      option.textContent = opt.label;
      input.appendChild(option);
    });
  } else {
    input = document.createElement("input");
    input.type = "number";
  }

  input.name = field.key;
  input.required = true;

  form.appendChild(label);
  form.appendChild(input);
});

// Add buttons
["RF", "XGB", "Ensemble"].forEach(model => {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.innerText = `Predict with ${model}`;
  btn.addEventListener("click", () => submitForm(model.toLowerCase()));
  form.appendChild(btn);
});

async function submitForm(model) {
  feedback.innerHTML = "";
  const data = {};
  const formData = new FormData(form);
  formData.forEach((value, key) => data[key] = value);

  try {
    const res = await fetch(`/predict_${model}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data)
    });

    if (!res.ok) throw new Error(`Prediction failed (${model})`);

    const result = await res.json();
    feedback.innerHTML = `<div class="result">${result.prediction}</div>`;
  } catch (err) {
    feedback.innerHTML = `<div class="error">${err.message}</div>`;
  }
}

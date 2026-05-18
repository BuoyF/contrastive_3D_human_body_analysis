import argparse
import os
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True, help="Ob.xlsx")
    ap.add_argument("--mesh_dir", required=True, help="Obstetrics/objs")
    ap.add_argument("--out_csv", default="ob_dataset.csv")
    args = ap.parse_args()

    df = pd.read_excel(args.xlsx)
    df["Subject_ID"] = df["Subject_ID"].ffill()
    df["clean_id"] = df["Subject_ID"].astype(str).str.extract(r"(\d+)").ffill().astype(str)

    # delivery_type: 1 if starts with 'c' (c-section)
    df["delivery_type"] = df["Delivery type"].astype(str).str.lower().str.startswith("c").astype(int)
    if "Head Circumference (cm)" in df.columns:
        df["Head Circumference (cm)"] = pd.to_numeric(df["Head Circumference (cm)"], errors="coerce")

    subject_rows = df.groupby("clean_id")

    mesh_records = []
    for fname in os.listdir(args.mesh_dir):
        if not fname.lower().endswith(".obj"):
            continue
        parts = fname.replace("-", "_").replace(".obj", "").split("_")
        if len(parts) < 1:
            continue
        try:
            subject_id = str(int(parts[0]))  # strip leading zeros
            mesh_path = os.path.join(args.mesh_dir, fname)
            mtime = os.path.getmtime(mesh_path)
            mesh_records.append({
                "file": fname,
                "subject_id": subject_id,
                "mesh_path": mesh_path,
                "timestamp": datetime.fromtimestamp(mtime),
                "delivery_type": int(df.loc[df["clean_id"] == subject_id, "delivery_type"].iloc[0]) if not df[df["clean_id"] == subject_id].empty else 0,
            })
        except Exception as e:
            print(f"Skipping file {fname}: {e}")

    mesh_df = pd.DataFrame(mesh_records)

    final_rows = []
    for subject_id, rows in tqdm(subject_rows, desc="Subjects"):
        rows = rows.reset_index(drop=True)
        header = rows.iloc[0]

        try:
            age = float(header["Age"])
            sex = 1.0 if str(header.get("Sex", "female")).lower() in ["1", "f", "female"] else 0.0
            race_raw = str(header.get("Race", "")).strip().lower()
            race_mapping = {
                "caucasian": 1.0, "white": 1.0,
                "hispanic": 2.0,
                "asian": 3.0, "other": 3.0,
                "african american": 4.0,
            }
            race = float(race_mapping.get(race_raw, 0.0))
            height = float(header["height (m)"])
            prev_csec = 0 if str(header.get("Previous C-section", "")).strip().lower() in ["n", "no"] else 1
            chronic = 0 if str(header.get("Pre-gravid chronic disease", "")).strip().lower() in ["n", "no"] else 1
            hx_comp = 0 if str(header.get("History of pregnancy complications", "")).strip().lower() in ["n", "no"] else 1
            head_circ = header.get("Head Circumference (cm)", np.nan)  # ✅ correct: subject-level
        except Exception as e:
            print(f"Skipping subject {subject_id}: {e}")
            continue

        appointments = rows[rows["Date_of_appointments"].notna()].copy()
        appointments["Date_of_appointments"] = pd.to_datetime(appointments["Date_of_appointments"], errors="coerce")

        subj_meshes = mesh_df[mesh_df["subject_id"] == subject_id]
        for _, mesh_row in subj_meshes.iterrows():
            mesh_time = mesh_row["timestamp"]
            diffs = (appointments["Date_of_appointments"] - mesh_time).abs()
            if len(diffs) == 0 or diffs.isnull().all():
                continue

            best_idx = diffs.idxmin()
            best_appt = appointments.loc[best_idx]

            try:
                weight = float(best_appt["weight (kg)"])
                appt_date = best_appt["Date_of_appointments"]

                ga_col = "Gestitional_age" if "Gestitional_age" in best_appt else "Gestational_age"
                ga_str = str(best_appt.get(ga_col, "") or "")
                if "," in ga_str:
                    gestational_age = int(ga_str.split(",")[0])
                elif ga_str.replace(".", "", 1).isdigit():
                    gestational_age = int(float(ga_str))
                else:
                    gestational_age = -1
            except Exception as e:
                print(f"Skipping mesh for subject {subject_id}: {e}")
                continue

            final_rows.append({
                "Subject_ID": subject_id,
                "Age": age,
                "Sex": sex,
                "Race": race,
                "Height": height,
                "Previous_C_Section": prev_csec,
                "Chronic_Disease": chronic,
                "Pregnancy_History": hx_comp,
                "Weight": weight,
                "Appointment_Date": appt_date.strftime("%Y-%m-%d") if pd.notnull(appt_date) else "",
                "Mesh_Path": mesh_row["mesh_path"].replace("\\", "/"),
                "Gestational_Age": gestational_age,
                "Sample_ID": f"{subject_id}_{mesh_row['file']}",
                "Delivery": int(mesh_row["delivery_type"]),
                "head_circumference": float(head_circ) if pd.notnull(head_circ) else np.nan,
            })

    out = pd.DataFrame(final_rows)
    out.to_csv(args.out_csv, index=False)
    print(f"Saved dataset with {len(out)} matched samples to {args.out_csv}")


if __name__ == "__main__":
    main()

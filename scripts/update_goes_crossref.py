"""Update New Flare Events.xlsx with GOES cross-reference data."""
import pandas as pd

df = pd.read_excel("New Flare Events.xlsx", sheet_name="Flare Events")

# Convert columns that may be read as numeric to string
for col in ["Flare_Onset_Time_UTC", "Flare_Onset_Date_UTC", "Onset_Time_Source", "Confidence_Status"]:
    df[col] = df[col].astype(str).replace("nan", "")


# Event SUIT_FLR_20251206_02 and _03 correspond to the M8.1 flare from AR 4299
# Peak at 20:39 UTC on Dec 6, 2025
dec6_mask = df["Event_ID"].isin(["SUIT_FLR_20251206_02", "SUIT_FLR_20251206_03"])
df.loc[dec6_mask, "Flare_Onset_Date_UTC"] = "2025-12-06"
df.loc[dec6_mask, "Flare_Onset_Time_UTC"] = "20:39:00"
df.loc[dec6_mask, "Onset_Time_Source"] = (
    "NOAA GOES-18/19 XRS (Class M8.1, AR 4299, beta-delta config), "
    + df.loc[dec6_mask, "Onset_Time_Source"]
)
df.loc[dec6_mask, "Confidence_Status"] = "Confirmed Flare Observation (GOES M8.1 cross-referenced)"

# Event SUIT_FLR_20251209_01 - Dec 9 at 02:00-02:38 UTC
# The X1.1 flare was on Dec 8 at 05:01 UTC from AR 4298
# This Dec 9 observation is ~21h after the X1.1 peak
dec9_mask = df["Event_ID"] == "SUIT_FLR_20251209_01"
df.loc[dec9_mask, "Flare_Onset_Date_UTC"] = "2025-12-09"
df.loc[dec9_mask, "Flare_Onset_Time_UTC"] = "~02:00:00 (ongoing activity)"
df.loc[dec9_mask, "Onset_Time_Source"] = (
    "SUIT Payload Flag (FLR_TRIG=5757483), "
    "Post X1.1 (Dec 8 05:01 UTC AR 4298) activity period"
)
df.loc[dec9_mask, "Confidence_Status"] = (
    "Candidate Flare Observation (Payload-flagged, post-X1.1 activity)"
)

# Save back
log_df = pd.read_excel("New Flare Events.xlsx", sheet_name="Processing Log")
with pd.ExcelWriter("New Flare Events.xlsx", engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Flare Events", index=False)
    log_df.to_excel(writer, sheet_name="Processing Log", index=False)

print("Updated New Flare Events.xlsx with GOES cross-reference")
print("Events:")
for eid in sorted(df["Event_ID"].unique()):
    sub = df[df["Event_ID"] == eid]
    print(f"  {eid}: {len(sub)} frames")
    onset_d = sub["Flare_Onset_Date_UTC"].iloc[0]
    onset_t = sub["Flare_Onset_Time_UTC"].iloc[0]
    status = sub["Confidence_Status"].iloc[0]
    print(f"    Onset: {onset_d} {onset_t}")
    print(f"    Status: {status}")

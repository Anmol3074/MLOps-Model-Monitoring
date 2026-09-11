import subprocess

# Run all monitoring stages
steps = [
    "python scripts/workflow.py",
    "python scripts/monitoring_report.py",
    "python scripts/monitor_api.py",
    "python scripts/prediction_monitor.py",
    "python scripts/alert_summary.py"
]

for step in steps:
    print(f"\nRunning: {step}")
    subprocess.run(step, shell=True, check=True)

print("\nFull MLOps monitoring pipeline completed.")
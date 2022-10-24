# if no arguments are given, the script will submit all jobs
if [ $# -eq 0 ]; then
    echo "No arguments given, submit all jobs"
    amlt run --upload-data submit_hai6.yaml
else
    job=$1
    echo "submit the job $job"
    amlt run --upload-data submit_hai6.yaml :$job
fi
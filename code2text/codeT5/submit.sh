# if no arguments are given, the script will submit all jobs
if [ $# -eq 0 ]; then
    echo "No arguments given, submit all jobs"
    amlt run --upload-data submit.yaml --preemptible
else
    job=$1
    echo "submit the job $job"
    amlt run --upload-data submit.yaml :$job --preemptible
fi
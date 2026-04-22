pipeline {
    agent {
        docker {
            image 'jager72/ium:0.1.5'
            args '-u root'
        }
    }
    parameters {
        string(name: 'CUT_OFF', defaultValue: '', description: 'Limit dataset size')
        string(name: 'INCLUDE_CONFUSION_MATRIX', defaultValue: '', description: 'Add Confustion Matrix to prediction (Boolean)')
    }
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        stage('Run Preparation') {
            steps {
                sh "mkdir -p artifacts"
                
                script {
                    def cmd = "uv run python src/prepareData.py"
                    if (params.CUT_OFF?.trim()) {
                        cmd += " --cut-off=${params.CUT_OFF}"
                    }
                    sh cmd
                }
            }
        }

        stage('Archive Prepared Data') {
            steps {
                archiveArtifacts artifacts: 'artifacts/train.csv, artifacts/test.csv, artifacts/eval.csv', fingerprint: true
            }
        }

        stage('Run Train') {
            steps {
                script {

                    sh "mlflow server \
                        --backend-store-uri sqlite:///mlflow.db \
                        --default-artifact-root ./mlruns \
                        --host 0.0.0.0 \
                        --port 8080 \
                        > mlflow.log 2>&1 &"

                    sh "uv run python src/train.py"

                    def runId = sh(
                        script: "cat artifacts/run_id.txt",
                        returnStdout: true
                    ).trim()

                    def cmd = "uv run python src/predict.py"
                    if (params.INCLUDE_CONFUSION_MATRIX?.trim()) {
                        cmd += " --include-confusion-matrix=${params.INCLUDE_CONFUSION_MATRIX}"
                    }

                    cmd += " --model-uri runs:/${runId}/model"
                    sh cmd
                }
            }
        }

        stage('Archive Model And Predictions') {
            steps {
                archiveArtifacts artifacts: 'artifacts/savePred.txt, artifacts/starbucks_model.pth, artifacts/predictionsMetrics.txt,artifacts/run_id.txt,mlruns/**', fingerprint: true
            }
        }
    }
}
## Project overview

This project trains or loads a pre-trained regression model (nifty_ridge_model.pkl) that, given features derived from historical market data and a requested future date, returns a predicted NIFTY50 value. The model is packaged and deployed as a serverless API (AWS Lambda). Two deployment methods are supported so you can choose according to packaging needs and dependency size:

Lambda Container Image — larger dependencies, custom runtime control. Image stored in ECR, deployed as Lambda function from container.

An Application Load Balancer (ALB) (public) sits in front of the Lambda and provides a stable HTTPS endpoint that Salesforce can call. The Lambda can run inside private subnets (for security) while ALB resides in public subnets and forwards requests to the Lambda target.

Model artifact and any required static data can be stored in S3. Logs, metrics, and traces flow to CloudWatch and X-Ray.

## URL

```bash
# For Predictio - GET Request
http://alb-lambda-integration-01-1423243074.ap-south-1.elb.amazonaws.com/predict

# For Health Check
http://alb-lambda-integration-01-1423243074.ap-south-1.elb.amazonaws.com
```

## Deployment

Prereqs: AWS CLI configured, Docker, GitHub repo cloned, AWS account with appropriate permissions.

Lambda as Docker Image (recommended for heavy deps)

Dockerfile (example)

```bash
FROM public.ecr.aws/lambda/python:3.10


# Copy application
COPY app.py ${LAMBDA_TASK_ROOT}/
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
COPY nifty_ridge_model.pkl ${LAMBDA_TASK_ROOT}/


RUN pip3 install -r ${LAMBDA_TASK_ROOT}/requirements.txt --target ${LAMBDA_TASK_ROOT}


CMD ["app.handler"]
```

Build & push

```bash
# build
docker build -t nifty-lambda:latest .


# tag
aws ecr create-repository --repository-name nifty-lambda || true
REPO_URI=$(aws ecr describe-repositories --repository-names nifty-lambda --query 'repositories[0].repositoryUri' --output text)


docker tag nifty-lambda:latest ${REPO_URI}:v1


# login & push
aws ecr get-login-password | docker login --username AWS --password-stdin ${REPO_URI}
docker push ${REPO_URI}:v1
```
Create Lambda from image

+ Create Lambda function in console or via CLI specifying the ECR image URI.

+ Configure the Lambda to run in private subnets (if needed), attach IAM role allowing S3/SecretsManager.

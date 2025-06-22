# serializeImageData function
import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    key = event['s3_key']
    bucket = event['s3_bucket']
    s3.download_file(bucket, key, '/tmp/image.png')
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    return {
        'statusCode': 200,
        'body': {
            "s3_bucket": bucket,
            "s3_key": key,
            "image_data": image_data,
            "inferences": []
        }
    }

# imageClassifier function
import json
import base64
import boto3
import sagemaker
from sagemaker.serializers import IdentitySerializer

ENDPOINT = 'image-classification-2024-08-11-16-40-35-761'

def lambda_handler(event, context):
    image = base64.b64decode(event['body']['image_data'])
    predictor = sagemaker.predictor.Predictor(ENDPOINT)
    predictor.serializer = IdentitySerializer("image/png")
    inferences = predictor.predict(image)
    event["body"]["inferences"] = json.loads(inferences.decode('utf-8'))
    return {
        'statusCode': 200,
        'body': event["body"]
    }

# filterInferences function
import json

THRESHOLD = .85

def lambda_handler(event, context):
    inferences = event["body"]["inferences"]
    meets_threshold = any(i > THRESHOLD for i in inferences)
    if meets_threshold:
        return {
            'statusCode': 200,
            'body': event["body"]
        }
    else:
        raise Exception("THRESHOLD_CONFIDENCE_NOT_MET")
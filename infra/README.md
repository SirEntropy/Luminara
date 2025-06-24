# Luminara EC2 Terraform Infrastructure

This module provisions a secure EC2 instance to host the Luminara FastAPI service.

## Usage

Set your AWS credentials, then:

```sh
cd infra
terraform init
terraform apply -var="ami_id=<your-ubuntu-ami>" -auto-approve
```

- The FastAPI service will be available on port 80 of the EC2 public IP.
- SSH is enabled for debugging (port 22). For production, restrict the security group as needed.

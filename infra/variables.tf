variable "aws_region" {
  description = "AWS region to deploy resources in"
  type        = string
  default     = "us-west-1"
}

variable "ami_id" {
  description = "AMI ID for EC2 instance (Ubuntu 22.04 recommended)"
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.medium"
}

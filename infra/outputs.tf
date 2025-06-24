output "instance_id" {
  value = aws_instance.fastapi.id
}

output "public_ip" {
  value = aws_instance.fastapi.public_ip
}

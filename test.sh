# python transfer.py \
#     --option_unpool sum -a \
#     --content ./examples/content \
#     --style ./examples/style \
#     --content_segment ./examples/content_segment \
#     --style_segment ./examples/style_segment/ \
#     --output ./outputs/ \
#     --verbose \
#     --image_size 512

python transfer.py \
    --option_unpool cat5 -d \
    --content ./examples/content \
    --style ./examples/style \
    --output ./outputs/ \
    --verbose \
    --image_size 512

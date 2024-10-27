from flask import Flask, render_template, url_for
import os
from pathlib import Path

app = Flask(__name__, static_url_path='', static_folder='train')

def get_idiom_dirs():
    train_path = Path('train')
    return sorted([d for d in train_path.iterdir() if d.is_dir()])

def get_images_from_dir(dir_path):
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif'}
    return sorted([
        f.name for f in dir_path.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ])[:5]  # Limit to first 5 images

@app.route('/')
@app.route('/<int:idiom_index>')
def show_idiom(idiom_index=0):
    idiom_dirs = get_idiom_dirs()
    
    # Handle index bounds
    if idiom_index >= len(idiom_dirs):
        idiom_index = 0
    elif idiom_index < 0:
        idiom_index = len(idiom_dirs) - 1
    
    current_dir = idiom_dirs[idiom_index]
    idiom_name = current_dir.name
    images = get_images_from_dir(current_dir)
    
    return render_template('idiom.html',
                         idiom_name=idiom_name,
                         images=images,
                         current_index=idiom_index,
                         prev_index=(idiom_index - 1),
                         next_index=(idiom_index + 1))

if __name__ == '__main__':
    app.run(port=5000, debug=True)

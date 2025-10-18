import os
import json
import uuid
import base64
from flask import Flask, render_template, request, jsonify, send_from_directory
import requests
import google.generativeai as genai
from datetime import datetime, date, timedelta
from werkzeug.utils import secure_filename
import mimetypes
from pathlib import Path
import threading

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # Limit to 20MB uploads
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CONVERSATIONS_FOLDER'] = 'logs/conversations'
app.config['USAGE_LOGS_FOLDER'] = 'logs/usage'

# File access lock - prevents simultaneous read/writes to the same file
file_lock = threading.Lock()

# Ensure required folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CONVERSATIONS_FOLDER'], exist_ok=True)
os.makedirs(app.config['USAGE_LOGS_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    'image': ['png', 'jpg', 'jpeg', 'gif', 'webp'],
    'audio': ['mp3', 'wav', 'ogg', 'm4a'],
    'video': ['mp4', 'webm', 'mov', 'avi'],
    'document': ['pdf', 'txt', 'doc', 'docx']
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in sum(ALLOWED_EXTENSIONS.values(), [])

def get_file_type(filename):
    ext = filename.rsplit('.', 1)[1].lower()
    for file_type, extensions in ALLOWED_EXTENSIONS.items():
        if ext in extensions:
            return file_type
    return 'unknown'

# Load chats from filesystem
def load_chats():
    chats = {}
    
    # Read all chat files
    chat_files = os.listdir(app.config['CONVERSATIONS_FOLDER'])
    for filename in chat_files:
        if filename.endswith('.json'):
            chat_id = filename[:-5]  # Remove .json extension
            chat_path = os.path.join(app.config['CONVERSATIONS_FOLDER'], filename)
            
            try:
                with file_lock:
                    with open(chat_path, 'r', encoding='utf-8') as f:
                        chat_data = json.load(f)
                        chats[chat_id] = chat_data
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error loading chat {chat_id}: {str(e)}")
                continue
    
    return chats

# Get a specific chat by ID from filesystem
def get_chat(chat_id):
    chat_path = os.path.join(app.config['CONVERSATIONS_FOLDER'], f"{chat_id}.json")
    
    try:
        with file_lock:
            with open(chat_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None

# Save a chat to filesystem
def save_chat(chat_id, chat_data):
    chat_path = os.path.join(app.config['CONVERSATIONS_FOLDER'], f"{chat_id}.json")
    
    with file_lock:
        with open(chat_path, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, ensure_ascii=False, indent=2)

# Log API usage
def log_api_usage(api_key, model, usage_type, token_count=0, attachments=None):
    # Create a date-based filename for today's log using DD/MM/YYYY format
    today = datetime.now().strftime('%d-%m-%Y')
    log_filename = f"{today}.json"
    log_path = os.path.join(app.config['USAGE_LOGS_FOLDER'], log_filename)
    
    # Add entry details
    entry = {
        'timestamp': datetime.now().isoformat(),
        'api_key': api_key[-4:],  # Only store last 4 chars for privacy
        'model': model,
        'type': usage_type,  # e.g., 'message', 'regenerate', 'chat_name'
        'token_estimate': token_count,
        'has_attachments': bool(attachments),
        'attachment_types': [a.get('type', 'unknown') for a in (attachments or [])]
    }
    
    # Read existing log or create new one
    log_data = []
    
    try:
        with file_lock:
            if os.path.exists(log_path):
                with open(log_path, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
            
            # Append new entry
            log_data.append(entry)
            
            # Write back to file
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error logging API usage: {str(e)}")

# Estimate tokens for message
def estimate_tokens(content, attachments=None):
    # Base token estimation (approx. 1.3 tokens per word)
    if not content:
        content_tokens = 0
    else:
        words = content.split()
        content_tokens = int(len(words) * 1.3)
    
    # Add tokens for attachments
    attachment_tokens = 0
    if attachments:
        for attachment in attachments:
            attachment_type = attachment.get('type', '')
            
            if attachment_type.startswith('image'):
                # Each image is approximately 258 tokens
                attachment_tokens += 258
            elif attachment_type.startswith('audio'):
                # Audio is 32 tokens per second, estimate 30 seconds
                attachment_tokens += 32 * 30
            elif attachment_type.startswith('video'):
                # Video is ~300 tokens per second, estimate 10 seconds
                attachment_tokens += 300 * 10
    
    return content_tokens + attachment_tokens

# Track current chat
current_chat_id = None

@app.route('/')
def index():
    return render_template('index.html')

# Add a route to check if the application is running
@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        'status': 'ok',
        'message': 'Application is running',
        'version': '1.0.0'
    })

# API endpoint to log settings changes
@app.route('/api/settings/log', methods=['POST'])
def log_settings():
    data = request.json
    setting_type = data.get('type', 'unknown')
    setting_value = data.get('value', 'unknown')
    
    print(f"Settings changed: {setting_type} set to {setting_value}")
    
    return jsonify({"status": "logged"})

# API endpoint to upload a file
@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Get the proper MIME type using mimetypes library
        content_type, _ = mimetypes.guess_type(file_path)
        if not content_type:
            # Fallback for common image types if mimetypes fails
            ext = filename.rsplit('.', 1)[1].lower()
            if ext == 'jpg' or ext == 'jpeg':
                content_type = 'image/jpeg'
            elif ext == 'png':
                content_type = 'image/png'
            elif ext == 'gif':
                content_type = 'image/gif'
            elif ext == 'webp':
                content_type = 'image/webp'
            else:
                # Generic fallback based on file type
                file_type = get_file_type(filename)
                content_type = f"{file_type}/{ext}"
        
        file_type = get_file_type(filename)
        file_size = os.path.getsize(file_path)
        
        file_url = f"/uploads/{filename}"
        
        return jsonify({
            'file_url': file_url,
            'file_name': filename,
            'file_type': file_type,
            'mime_type': content_type,
            'file_size': file_size
        })
    
    return jsonify({'error': 'File type not allowed'}), 400

# API endpoint to access uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # Get the MIME type for the file
    mime_type, _ = mimetypes.guess_type(filename)
    if not mime_type:
        # Fallback for common types
        ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        if ext in ['jpg', 'jpeg']:
            mime_type = 'image/jpeg'
        elif ext == 'png':
            mime_type = 'image/png'
        elif ext == 'gif':
            mime_type = 'image/gif'
        elif ext == 'webp':
            mime_type = 'image/webp'
        elif ext in ['mp3', 'wav', 'ogg']:
            mime_type = f'audio/{ext}'
        elif ext in ['mp4', 'webm', 'mov']:
            mime_type = f'video/{ext}'
        else:
            mime_type = 'application/octet-stream'
    
    return send_from_directory(
        app.config['UPLOAD_FOLDER'], 
        filename,
        mimetype=mime_type
    )

# API endpoint to get all chats
@app.route('/api/chats', methods=['GET'])
def get_all_chats():
    chats_data = load_chats()
    # Filter out temporary chats
    permanent_chats = {chat_id: chat_data for chat_id, chat_data in chats_data.items() 
                      if not chat_data.get('is_temporary', False)}
    # Convert dictionary to list for the response
    return jsonify(list(permanent_chats.values()))

# API endpoint to create a new chat
@app.route('/api/chats', methods=['POST'])
def create_chat():
    global current_chat_id
    chat_id = str(uuid.uuid4())
    
    chat_data = {
        'id': chat_id,
        'name': 'New Chat',
        'messages': [],
        'created_at': datetime.now().isoformat(),
        'is_temporary': True  # Flag to indicate this is a temporary chat
    }
    
    # For temporary chats, we don't save to the database yet
    # They will be saved when the first message is added
    
    current_chat_id = chat_id
    return jsonify(chat_data)

# API endpoint to get a specific chat
@app.route('/api/chats/<chat_id>', methods=['GET'])
def get_specific_chat(chat_id):
    chat_data = get_chat(chat_id)
    
    if chat_data:
        return jsonify(chat_data)
    return jsonify({'error': 'Chat not found'}), 404

# API endpoint to update a chat name
@app.route('/api/chats/<chat_id>/name', methods=['PUT'])
def rename_chat(chat_id):
    try:
        data = request.get_json()
        new_name = data.get('name')
        
        if not new_name:
            return jsonify({'error': 'Name is required'}), 400
            
        # Get the chat file path
        chat_path = os.path.join(app.config['CONVERSATIONS_FOLDER'], f"{chat_id}.json")
        
        # Check if chat exists
        if not os.path.exists(chat_path):
            return jsonify({'error': 'Chat not found'}), 404
            
        # Read the chat data
        chat_data = get_chat(chat_id)
        if not chat_data:
            return jsonify({'error': 'Chat not found'}), 404
            
        # Update the chat name
        chat_data['name'] = new_name
        
        # Save the updated chat
        save_chat(chat_id, chat_data)
        
        return jsonify({'success': True, 'name': new_name})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API endpoint to delete a chat
@app.route('/api/chats/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    try:
        # Get the chat file path
        chat_path = os.path.join(app.config['CONVERSATIONS_FOLDER'], f"{chat_id}.json")
        
        # Check if chat exists
        if not os.path.exists(chat_path):
            return jsonify({'error': 'Chat not found'}), 404
            
        # First, read the chat data to find associated files
        attachments_to_clean = []
        try:
            with file_lock:
                with open(chat_path, 'r', encoding='utf-8') as f:
                    chat_data = json.load(f)
                    
                # Find all attachments in messages
                for message in chat_data.get('messages', []):
                    if 'attachments' in message:
                        for attachment in message['attachments']:
                            if 'name' in attachment:
                                attachments_to_clean.append(attachment['name'])
        except Exception as e:
            print(f"Error reading chat data: {str(e)}")
            # Continue with deletion even if we can't read the chat data
            
        # Now delete the chat file
        with file_lock:
            os.remove(chat_path)
            
        # Clean up any associated uploaded files
        for filename in attachments_to_clean:
            try:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                # Log the error but don't fail the deletion
                print(f"Error deleting attachment {filename}: {str(e)}")
        
        return jsonify({'success': True, 'message': 'Chat deleted successfully'})
    except Exception as e:
        print(f"Error deleting chat: {str(e)}")
        return jsonify({'error': f'Failed to delete chat: {str(e)}'}), 500

# API endpoint to send a message and get a response
@app.route('/api/chats/<chat_id>/messages', methods=['POST'])
def send_message(chat_id):
    try:
        # Get the message data
        data = request.get_json()
        user_message = data.get('message', '')
        api_key = data.get('api_key')
        model = data.get('model', 'gemini-1.5-flash')
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('max_tokens', 2048)
        top_p = data.get('top_p', 0.95)
        attachments = data.get('attachments', [])
        
        # Validate API key
        if not api_key:
            return jsonify({'error': 'API key is required'}), 400
            
        # Get the chat data
        chat_data = get_chat(chat_id)
        
        # If chat doesn't exist, create it
        if not chat_data:
            chat_data = {
                'id': chat_id,
                'name': 'New Chat',
                'messages': [],
                'created_at': datetime.now().isoformat(),
                'is_temporary': False  # This is now a permanent chat
            }
            # Save the new chat
            save_chat(chat_id, chat_data)
        
        # If this was a temporary chat, make it permanent
        if chat_data.get('is_temporary', False):
            chat_data['is_temporary'] = False
            # Save the chat to make it permanent
            save_chat(chat_id, chat_data)
        
        # Add user message to chat
        if 'messages' not in chat_data:
            chat_data['messages'] = []
            
        user_message_obj = {
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add attachments to the message if any
        if attachments:
            user_message_obj['attachments'] = attachments
            
        chat_data['messages'].append(user_message_obj)
        
        # Save the updated chat
        save_chat(chat_id, chat_data)
        
        # Call Gemini API with validated parameters
        try:
            response = call_gemini_api(
                message=user_message, 
                api_key=api_key, 
                model_name=model, 
                chat_history=chat_data.get('messages', []), 
                temperature=temperature, 
                max_tokens=max_tokens, 
                top_p=top_p, 
                attachments=attachments
            )
        except Exception as api_error:
            print(f"Error calling Gemini API: {str(api_error)}")
            return jsonify({'error': f"API Error: {str(api_error)}"}), 500
        
        # Add AI response to chat
        ai_message_obj = {
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now().isoformat()
        }
        
        chat_data['messages'].append(ai_message_obj)
        
        # Update chat name based on conversation if it's still "New Chat" and we have at least 2 exchanges
        if chat_data['name'] == 'New Chat' and len(chat_data['messages']) >= 4:  # 2 user messages + 2 AI responses
            try:
                # Generate a name based on the conversation
                chat_name = generate_chat_name(chat_data['messages'], api_key, model)
                if chat_name:
                    chat_data['name'] = chat_name
            except Exception as name_error:
                print(f"Error generating chat name: {str(name_error)}")
                # Fallback to using the first message if name generation fails
                if user_message:
                    chat_name = user_message[:30]
                    if len(user_message) > 30:
                        chat_name += '...'
                    chat_data['name'] = chat_name
        
        # Save the updated chat again with the AI response
        save_chat(chat_id, chat_data)
        
        return jsonify({
            'message': ai_message_obj,
            'chat': chat_data
        })
    except Exception as e:
        print(f"Error in send_message: {str(e)}")
        return jsonify({'error': str(e)}), 500

def call_gemini_api(message, api_key, model_name, chat_history=None, temperature=0.7, max_tokens=2048, top_p=0.95, attachments=None):
    # Configure the Google Generative AI with the provided API key
    genai.configure(api_key=api_key)
    
    # Select the model
    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        # Fallback to a known model if the requested one is not available
        print(f"Error with model {model_name}: {str(e)}. Falling back to gemini-1.5-pro")
        model = genai.GenerativeModel("gemini-1.5-pro")
    
    # Validate and sanitize parameters
    try:
        # Temperature must be between 0.0 and 2.0
        temperature = float(temperature)
        if temperature < 0.0 or temperature > 2.0:
            print(f"Temperature {temperature} out of range [0.0, 2.0]. Using default 0.7")
            temperature = 0.7
    except (TypeError, ValueError):
        print(f"Invalid temperature value: {temperature}. Using default 0.7")
        temperature = 0.7
        
    try:
        # Top_p must be between 0.0 and 1.0
        top_p = float(top_p)
        if top_p < 0.0 or top_p > 1.0:
            print(f"Top_p {top_p} out of range [0.0, 1.0]. Using default 0.95")
            top_p = 0.95
    except (TypeError, ValueError):
        print(f"Invalid top_p value: {top_p}. Using default 0.95")
        top_p = 0.95
        
    try:
        # Max tokens must be positive
        max_tokens = int(max_tokens)
        if max_tokens <= 0:
            print(f"Max tokens {max_tokens} must be positive. Using default 2048")
            max_tokens = 2048
    except (TypeError, ValueError):
        print(f"Invalid max_tokens value: {max_tokens}. Using default 2048")
        max_tokens = 2048
    
    # Configure generation parameters with validated values
    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=40,
        max_output_tokens=max_tokens,
    )
    
    # Prepare the chat history
    chat = model.start_chat(history=[])
    
    # Ensure chat_history is a list before trying to use len()
    if chat_history is not None and isinstance(chat_history, list):
        # Format previous messages for the chat history
        formatted_history = []
        
        # Only include up to the 10 most recent messages to avoid token limits
        if len(chat_history) > 10:
            relevant_history = chat_history[-10:-1]
        else:
            relevant_history = chat_history[:-1] if chat_history else []
        
        for msg in relevant_history:
            role = "user" if msg['role'] == 'user' else "model"
            content = msg['content']
            
            if role == "user":
                formatted_history.append({"role": "user", "parts": [content]})
            else:
                formatted_history.append({"role": "model", "parts": [content]})
        
        # Update the chat history
        chat._history = formatted_history
    
    # Prepare the message parts
    message_parts = []
    has_multimodal = False
    
    # Add any attachments
    if attachments and len(attachments) > 0:
        for attachment in attachments:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(attachment['name']))
            # Use the mime_type from attachment if available, otherwise get it from the file
            mime_type = attachment.get('mime_type')
            if not mime_type:
                mime_type, _ = mimetypes.guess_type(file_path)
                if not mime_type:
                    # Fallback based on type
                    file_type = attachment.get('type', 'unknown')
                    ext = attachment['name'].rsplit('.', 1)[1].lower() if '.' in attachment['name'] else 'unknown'
                    mime_type = f"{file_type}/{ext}"
            
            try:
                # Handle image, audio, or video attachments
                if mime_type and (mime_type.startswith('image') or 
                                 mime_type.startswith('audio') or 
                                 mime_type.startswith('video')):
                    # For multimedia, read and encode as base64
                    with open(file_path, "rb") as f:
                        file_data = f.read()
                    
                    # Check file size for video (warn if it might be too large)
                    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    if mime_type.startswith('video') and file_size_mb > 100:
                        print(f"Warning: Large video file ({file_size_mb:.1f} MB) may consume many tokens")
                    
                    # Create a dictionary with mime_type and data fields
                    data_part = {
                        "mime_type": mime_type,
                        "data": base64.b64encode(file_data).decode('utf-8')
                    }
                    message_parts.append(data_part)
                    has_multimodal = True
                    print(f"Added {mime_type} attachment: {attachment['name']}")
                
                else:
                    # For other files, mention the attachment 
                    message_parts.append(f"[Attached file: {attachment['name']}]")
            
            except Exception as e:
                print(f"Error processing attachment {attachment['name']}: {str(e)}")
                message_parts.append(f"[Error processing attachment: {attachment['name']}]")
    
    # Add text message if provided
    if message:
        message_parts.append(message)
    
    # If no message parts, add a generic prompt
    if not message_parts:
        message_parts.append("Please analyze the attached content.")
    
    # Send the message and get the response
    try:
        # Define safety settings
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
        
        # For multimodal inputs, format the request properly
        if has_multimodal:
            # Create proper parts array for multimodal request
            parts = []
            for part in message_parts:
                if isinstance(part, dict) and 'mime_type' in part:  # Multimodal part
                    parts.append({
                        "inline_data": {
                            "mime_type": part["mime_type"],
                            "data": part["data"]
                        }
                    })
                else:  # Text part
                    parts.append({"text": str(part)})
            
            response = model.generate_content(
                parts,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            return response.text
        else:
            # For text-only inputs, join all parts into a single text message
            text_message = " ".join(str(part) for part in message_parts)
            response = chat.send_message(
                text_message,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            return response.text
    except Exception as e:
        print(f"Error in API call: {str(e)}")
        raise Exception(f"Error processing request: {str(e)}")

def generate_chat_name(messages, api_key, model_name):
    # Extract the first exchange to create a name
    conversation = "\n".join([msg['content'] for msg in messages[:2]])
    
    # Configure the Google Generative AI with the provided API key
    genai.configure(api_key=api_key)
    
    try:
        model = genai.GenerativeModel(model_name)
    except:
        # Fallback to a known model if the requested one is not available
        model = genai.GenerativeModel("gemini-1.5-pro")
    
    prompt = f"Based on this conversation, provide a very concise title (max 4-5 words) that describes the main topic. Only respond with the title, no explanation:\n\n{conversation}"
    
    # Use lower temperature for more consistent chat names
    generation_config = genai.types.GenerationConfig(
        temperature=0.2,
        top_p=0.95,
        top_k=40,
        max_output_tokens=20,
    )
    
    try:
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        chat_name = response.text
        # Clean and limit the name
        chat_name = chat_name.strip().strip('"\'').replace('\n', ' ')
        return chat_name[:30]  # Limit to 30 characters
    except Exception as e:
        print(f"Error generating chat name: {str(e)}")
    
    return None

# API endpoint to get usage logs
@app.route('/api/usage', methods=['GET'])
def get_usage_logs():
    days = request.args.get('days', default=7, type=int)
    logs = fetch_usage_logs(days)
    return jsonify(logs)

def fetch_usage_logs(days=7):
    """Get API usage logs for the specified number of days."""
    logs = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    current_date = start_date
    while current_date <= end_date:
        # Use DD-MM-YYYY format for file names
        date_str = current_date.strftime('%d-%m-%Y')
        log_file = os.path.join(app.config['USAGE_LOGS_FOLDER'], f'{date_str}.json')
        
        if os.path.exists(log_file):
            with file_lock:
                with open(log_file, 'r', encoding='utf-8') as f:
                    daily_logs = json.load(f)
                    
                    # Calculate summary for this day
                    total_requests = len(daily_logs)
                    total_tokens = sum(entry.get('token_estimate', 0) for entry in daily_logs)
                    
                    # Count by type
                    requests_by_type = {}
                    for entry in daily_logs:
                        req_type = entry.get('type', 'unknown')
                        requests_by_type[req_type] = requests_by_type.get(req_type, 0) + 1
                    
                    # Count by model
                    requests_by_model = {}
                    for entry in daily_logs:
                        model = entry.get('model', 'unknown')
                        requests_by_model[model] = requests_by_model.get(model, 0) + 1
                    
                    # Format the date in DD/MM/YYYY format for display
                    formatted_date = current_date.strftime('%d/%m/%Y')
                    
                    logs.append({
                        'date': formatted_date,
                        'total_requests': total_requests,
                        'total_tokens': total_tokens,
                        'by_type': requests_by_type,
                        'by_model': requests_by_model,
                        'entries': daily_logs
                    })
        
        current_date += timedelta(days=1)
    
    # Sort logs by date in descending order (most recent first)
    logs.sort(key=lambda x: datetime.strptime(x['date'], '%d/%m/%Y'), reverse=True)
    return logs

if __name__ == '__main__':
    app.run(debug=True)

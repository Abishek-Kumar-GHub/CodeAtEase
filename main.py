from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import httpx
import os
from datetime import datetime, timedelta
import jwt
from passlib.context import CryptContext
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import base64

load_dotenv()

templates = Jinja2Templates(directory="templates")

# FastAPI App
app = FastAPI(
    title="CodeAtEase API",
    description="AI-powered code editor with GitHub integration",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
SECRET_KEY = os.environ.get("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440

GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")
GITHUB_REDIRECT_URI = os.getenv("GITHUB_REDIRECT_URI", "http://localhost:8000/auth/github/callback")

HF_TOKEN = os.getenv("HF_TOKEN")

# Security
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# In-memory storage
users_db: Dict[int, Dict] = {}
tokens_db: Dict[str, int] = {}

# Initialize Hugging Face client
hf_client = InferenceClient(
    provider="featherless-ai",
    api_key=HF_TOKEN,
) if HF_TOKEN else None

# ==================== MODELS ====================

class User(BaseModel):
    id: int
    username: str
    name: str
    email: Optional[str] = None
    avatar: str

class AnalyzeRequest(BaseModel):
    prompt: str
    selectedCode: Optional[str] = ""
    currentFile: Optional[Dict[str, Any]] = {}
    repository: Optional[List[Dict[str, Any]]] = []

# ==================== AUTHENTICATION ====================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    if "sub" in to_encode and isinstance(to_encode["sub"], int):
        to_encode["sub"] = str(to_encode["sub"])
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    if not token:
        raise HTTPException(status_code=401, detail="Token missing")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id_str = payload.get("sub")
        if user_id_str is None:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        user_id = int(user_id_str)
        if user_id not in users_db:
            raise HTTPException(status_code=401, detail="User not found")
        return users_db[user_id]
    except jwt.DecodeError:
        raise HTTPException(status_code=401, detail="Token decode error")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid user ID format")

# ==================== TEMPLATE ROUTES ====================

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/repo.html", response_class=HTMLResponse)
async def repo_page(request: Request):
    return templates.TemplateResponse("repo.html", {"request": request})

@app.get("/aipage.html", response_class=HTMLResponse)
async def ai_page(request: Request):
    return templates.TemplateResponse("aipage.html", {"request": request})

# ==================== AUTH ROUTES ====================

@app.get("/auth/github")
async def github_login():
    """Redirect to GitHub OAuth"""
    auth_url = (
        f"https://github.com/login/oauth/authorize?"
        f"client_id={GITHUB_CLIENT_ID}&"
        f"redirect_uri={GITHUB_REDIRECT_URI}&"
        f"scope=repo,user"
    )
    return RedirectResponse(auth_url)

@app.get("/auth/github/callback")
async def github_callback(code: str):
    """Handle GitHub OAuth callback"""
    if not code:
        raise HTTPException(status_code=400, detail="No code provided")
    
    async with httpx.AsyncClient() as client:
        token_response = await client.post(
            "https://github.com/login/oauth/access_token",
            headers={"Accept": "application/json"},
            data={
                "client_id": GITHUB_CLIENT_ID,
                "client_secret": GITHUB_CLIENT_SECRET,
                "code": code,
                "redirect_uri": GITHUB_REDIRECT_URI
            }
        )
        
        token_data = token_response.json()
        github_access_token = token_data.get("access_token")
        
        if not github_access_token:
            raise HTTPException(status_code=400, detail="Failed to get access token")
        
        user_response = await client.get(
            "https://api.github.com/user",
            headers={
                "Authorization": f"token {github_access_token}",
                "Accept": "application/json"
            }
        )
        
        user_data = user_response.json()
        user_id = user_data["id"]
        
        users_db[user_id] = {
            "id": user_id,
            "username": user_data["login"],
            "name": user_data.get("name", user_data["login"]),
            "email": user_data.get("email", ""),
            "avatar": user_data["login"][:2].upper(),
            "github_token": github_access_token,
            "created_at": datetime.now().isoformat()
        }
        
        jwt_token = create_access_token(data={"sub": user_id})
        tokens_db[jwt_token] = user_id
        
        redirect_url = f"http://127.0.0.1:8000/repo.html?access_token={jwt_token}"
        return RedirectResponse(redirect_url)

@app.get("/auth/user", response_model=User)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current authenticated user"""
    return User(
        id=current_user["id"],
        username=current_user["username"],
        name=current_user["name"],
        email=current_user.get("email", ""),
        avatar=current_user["avatar"]
    )

@app.post("/auth/logout")
async def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Logout user"""
    token = credentials.credentials
    if token in tokens_db:
        del tokens_db[token]
    return {"message": "Logged out successfully"}

# ==================== REPOSITORY ROUTES ====================

@app.get("/api/repositories")
async def get_repositories(current_user: dict = Depends(get_current_user)):
    """Get all repositories for authenticated user"""
    github_token = current_user["github_token"]
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            all_repos = []
            page = 1
            per_page = 100
            
            while True:
                response = await client.get(
                    "https://api.github.com/user/repos",
                    headers={
                        "Authorization": f"token {github_token}",
                        "Accept": "application/vnd.github.v3+json"
                    },
                    params={
                        "per_page": per_page,
                        "page": page,
                        "sort": "updated",
                        "affiliation": "owner,collaborator,organization_member"
                    }
                )
                
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"GitHub API error: {response.text}"
                    )
                
                repos = response.json()
                
                if not repos:
                    break
                
                all_repos.extend(repos)
                
                if len(repos) < per_page:
                    break
                
                page += 1
            
            repositories = [{
                "id": repo["id"],
                "name": repo["name"],
                "full_name": repo["full_name"],
                "owner": repo["owner"]["login"],
                "description": repo.get("description", ""),
                "private": repo["private"],
                "url": repo["html_url"],
                "clone_url": repo["clone_url"],
                "default_branch": repo.get("default_branch", "main"),
                "language": repo.get("language", ""),
                "stargazers_count": repo.get("stargazers_count", 0),
                "forks_count": repo.get("forks_count", 0),
                "updated_at": repo["updated_at"],
                "created_at": repo["created_at"],
                "size": repo.get("size", 0)
            } for repo in all_repos]
            
            return {"repositories": repositories, "total": len(repositories)}
        
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="GitHub API timeout")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch repositories: {str(e)}")

@app.get("/api/repository/tree/{owner}/{repo}")
async def get_repository_tree(
    owner: str,
    repo: str,
    current_user: dict = Depends(get_current_user)
):
    """Get complete repository file tree"""
    github_token = current_user["github_token"]
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Get repository info first
            repo_response = await client.get(
                f"https://api.github.com/repos/{owner}/{repo}",
                headers={
                    "Authorization": f"token {github_token}",
                    "Accept": "application/vnd.github.v3+json"
                }
            )
            
            if repo_response.status_code != 200:
                raise HTTPException(status_code=404, detail="Repository not found")
            
            repo_data = repo_response.json()
            default_branch = repo_data.get("default_branch", "main")
            
            # Get tree recursively
            tree_response = await client.get(
                f"https://api.github.com/repos/{owner}/{repo}/git/trees/{default_branch}?recursive=1",
                headers={
                    "Authorization": f"token {github_token}",
                    "Accept": "application/vnd.github.v3+json"
                }
            )
            
            if tree_response.status_code != 200:
                raise HTTPException(status_code=404, detail="Failed to fetch repository tree")
            
            tree_data = tree_response.json()
            
            # Build hierarchical structure
            file_tree = build_tree_structure(tree_data["tree"])
            
            return {
                "owner": owner,
                "repo": repo,
                "default_branch": default_branch,
                "tree": file_tree
            }
            
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="GitHub API timeout")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch repository tree: {str(e)}")

@app.get("/api/repository/file/{owner}/{repo}")
async def get_file_content(
    owner: str,
    repo: str,
    path: str,
    current_user: dict = Depends(get_current_user)
):
    """Get file content from repository"""
    github_token = current_user["github_token"]
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(
                f"https://api.github.com/repos/{owner}/{repo}/contents/{path}",
                headers={
                    "Authorization": f"token {github_token}",
                    "Accept": "application/vnd.github.v3+json"
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=404, detail="File not found")
            
            file_data = response.json()
            
            # Decode content
            try:
                content = base64.b64decode(file_data["content"]).decode("utf-8")
            except:
                content = "[Binary file - cannot display]"
            
            return {
                "path": file_data["path"],
                "name": file_data["name"],
                "content": content,
                "sha": file_data["sha"],
                "size": file_data["size"]
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch file: {str(e)}")

# ==================== AI ANALYSIS ====================

@app.post("/api/analyze")
async def analyze_code(
    request: AnalyzeRequest,
    current_user: dict = Depends(get_current_user)
):
    """Analyze code using AI"""
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    
    # Build AI prompt
    ai_prompt = f"""You are CatAI, an expert code assistant for the CodeAtEase platform.

User's Question: {request.prompt}

"""
    
    if request.currentFile and request.currentFile.get('path'):
        ai_prompt += f"""Current File: {request.currentFile.get('path')}

File Content:
```
{request.currentFile.get('content', '')[:2000]}
```

"""
    
    if request.selectedCode:
        ai_prompt += f"""Selected Code:
```
{request.selectedCode}
```

"""
    
    ai_prompt += """Please analyze the code and provide:
1. Any issues or bugs found
2. Recommended fixes with code examples
3. Best practices and improvements
4. Explanation of the solution

Be specific and provide actionable code snippets."""

    try:
        # Use Hugging Face if available
        if hf_client:
            response = await analyze_with_hf(ai_prompt)
        else:
            # Fallback mock response
            response = generate_mock_response(request)
        
        return {"response": response}
    
    except Exception as e:
        # Fallback to mock response on error
        response = generate_mock_response(request)
        return {"response": response}

async def analyze_with_hf(prompt: str) -> str:
    """Analyze code using Hugging Face LLaMA model"""
    try:
        result = hf_client.text_generation(
            prompt,
            model="meta-llama/Llama-3.1-8B",
            max_new_tokens=512,
            temperature=0.7,
        )
        
        if isinstance(result, str):
            return result
        elif isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "No response generated")
        elif isinstance(result, dict):
            return result.get("generated_text", "No response generated")
        
        return "No response generated"
    except Exception as e:
        print(f"HF API Error: {str(e)}")
        raise

def generate_mock_response(request: AnalyzeRequest) -> str:
    """Generate mock AI response for testing"""
    response = f"Analysis for: '{request.prompt}'\n\n"
    
    if request.currentFile and request.currentFile.get('path'):
        response += f"File: {request.currentFile.get('path')}\n\n"
    
    if request.selectedCode:
        response += f"Selected Code Analysis:\n{request.selectedCode[:200]}...\n\n"
    
    response += """✓ Code Quality: Good
✓ Best Practices: Consider adding error handling
✓ Performance: No major issues found
✓ Security: Looks safe

Recommendations:
- Add input validation
- Include error handling
- Add unit tests
- Consider code documentation"""
    
    return response

def build_tree_structure(items: List[Dict]) -> List[Dict]:
    """Build hierarchical tree structure from flat list"""
    tree = []
    path_dict = {}
    
    # Sort items to ensure folders come before files
    items_sorted = sorted(items, key=lambda x: (x['path'].count('/'), x['path']))
    
    for item in items_sorted:
        path = item['path']
        parts = path.split('/')
        
        node = {
            "name": parts[-1],
            "path": path,
            "type": "folder" if item['type'] == 'tree' else "file",
        }
        
        if item['type'] == 'tree':
            node["children"] = []
            node["expanded"] = False
        
        path_dict[path] = node
        
        if len(parts) == 1:
            tree.append(node)
        else:
            parent_path = '/'.join(parts[:-1])
            if parent_path in path_dict:
                if "children" not in path_dict[parent_path]:
                    path_dict[parent_path]["children"] = []
                path_dict[parent_path]["children"].append(node)
    
    return tree

# ==================== HEALTH CHECK ====================

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
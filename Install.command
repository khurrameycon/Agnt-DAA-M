#!/usr/bin/env bash

# My1ai Install.command
# Enhanced macOS Installer with dependency check and repair

set -e  # Exit on any error

# ==================================================
# Configuration
# ==================================================
APP_NAME="My1ai"
USER_HOME="$HOME"
TARGET_INSTALL_DIR="$USER_HOME/.my1ai/desktop-app"
TEMP_DIR="/tmp/my1ai_installer"
DOWNLOAD_DIR="$TEMP_DIR/downloads"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERNAL_DIR="$SCRIPT_DIR/_internal"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_check() {
    echo -e "${PURPLE}[CHECK]${NC} $1"
}

# Function to check if installation exists
check_existing_installation() {
    if [[ -d "$TARGET_INSTALL_DIR" ]] && [[ -d "$TARGET_INSTALL_DIR/.venv" ]] && [[ -f "$TARGET_INSTALL_DIR/main.py" ]]; then
        return 0  # Installation exists
    else
        return 1  # Installation doesn't exist
    fi
}

# Function to run dependency check and repair
run_dependency_check() {
    log_check "Running dependency check and repair..."
    
    cd "$TARGET_INSTALL_DIR"
    source .venv/bin/activate
    
    if [[ -z "$VIRTUAL_ENV" ]]; then
        log_error "Failed to activate virtual environment for dependency check"
        return 1
    fi
    
    log_info "Checking critical dependencies..."
    
    # Define critical dependencies that must work
    CRITICAL_DEPS=(
        "dotenv:python-dotenv"
        "PyQt6.QtCore:PyQt6"
        "requests:requests"
        "markdown:markdown"
        "certifi:certifi"
    )
    
    # Define important dependencies (app works without them but with limited features)
    IMPORTANT_DEPS=(
        "openai:openai"
        "groq:groq"
        "anthropic:anthropic"
        "transformers:transformers"
        "huggingface_hub:huggingface_hub"
        "gradio_client:gradio_client"
        "duckduckgo_search:duckduckgo_search"
        "smolagents:smolagents"
        "sentence_transformers:sentence_transformers"
        "PyPDF2:pypdf2"
        "google.genai:google-genai"
        "langchain:langchain"
        "chromadb:chromadb"
        "unidecode:unidecode"
    )
    
    # Define optional dependencies (nice to have)
    OPTIONAL_DEPS=(
        "peft:peft"
        "accelerate:accelerate"
        "datasets:datasets"
        "torch:torch"
        "selenium:selenium"
        "webdriver_manager:webdriver_manager"
        "bitsandbytes:bitsandbytes"
        "helium:helium"
        "yaml:pyyaml"
        "PIL:pillow"
        "tqdm:tqdm"
    )
    
    missing_critical=()
    missing_important=()
    missing_optional=()
    
    # Check critical dependencies
    log_info "Checking critical dependencies..."
    for dep in "${CRITICAL_DEPS[@]}"; do
        IFS=':' read -r import_name package_name <<< "$dep"
        if ! python -c "import $import_name" 2>/dev/null; then
            missing_critical+=("$package_name")
            log_warning "❌ Critical: $import_name missing"
        else
            log_success "✅ Critical: $import_name OK"
        fi
    done
    
    # Check important dependencies
    log_info "Checking important dependencies..."
    for dep in "${IMPORTANT_DEPS[@]}"; do
        IFS=':' read -r import_name package_name <<< "$dep"
        if ! python -c "import $import_name" 2>/dev/null; then
            missing_important+=("$package_name")
            log_warning "⚠️ Important: $import_name missing"
        else
            log_success "✅ Important: $import_name OK"
        fi
    done
    
    # Check optional dependencies
    log_info "Checking optional dependencies..."
    for dep in "${OPTIONAL_DEPS[@]}"; do
        IFS=':' read -r import_name package_name <<< "$dep"
        if ! python -c "import $import_name" 2>/dev/null; then
            missing_optional+=("$package_name")
            log_info "💡 Optional: $import_name missing"
        else
            log_success "✅ Optional: $import_name OK"
        fi
    done
    
    # Install missing packages
    if [[ ${#missing_critical[@]} -gt 0 ]]; then
        log_error "Critical dependencies missing: ${missing_critical[*]}"
        log_info "Installing critical dependencies..."
        for package in "${missing_critical[@]}"; do
            echo "Installing critical package: $package"
            pip install "$package" || log_error "Failed to install $package"
        done
    fi
    
    if [[ ${#missing_important[@]} -gt 0 ]]; then
        log_warning "Important dependencies missing: ${missing_important[*]}"
        log_info "Installing important dependencies..."
        for package in "${missing_important[@]}"; do
            echo "Installing important package: $package"
            pip install "$package" || log_warning "Failed to install $package"
        done
    fi
    
    if [[ ${#missing_optional[@]} -gt 0 ]]; then
        log_info "Optional dependencies missing: ${missing_optional[*]}"
        read -p "Do you want to install optional dependencies? (Y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            log_info "Installing optional dependencies..."
            for package in "${missing_optional[@]}"; do
                echo "Installing optional package: $package"
                pip install "$package" || log_info "Skipped $package (may not be compatible)"
            done
        fi
    fi
    
    # Final verification
    log_check "Running final verification..."
    python -c "
import sys
failed_critical = []

critical_imports = ['dotenv', 'PyQt6.QtCore', 'requests', 'markdown', 'certifi']

for module in critical_imports:
    try:
        __import__(module)
        print(f'✅ {module} verified')
    except ImportError:
        failed_critical.append(module)
        print(f'❌ {module} still missing')

if failed_critical:
    print(f'\\n❌ Critical dependencies still missing: {failed_critical}')
    print('The app may not work properly.')
    sys.exit(1)
else:
    print('\\n✅ All critical dependencies verified!')
    print('$APP_NAME should work properly now.')
"
    
    return $?
}

# ==================================================
# Main Menu System
# ==================================================
show_main_menu() {
    echo ""
    echo "🔧 $APP_NAME Installer & Manager"
    echo "================================"
    echo ""
    
    if check_existing_installation; then
        log_success "Existing installation found at: $TARGET_INSTALL_DIR"
        echo ""
        echo "Select an option:"
        echo "1) 🚀 Launch $APP_NAME"
        echo "2) 🔍 Check & Repair Dependencies"
        echo "3) 🔄 Update Installation (reinstall)"
        echo "4) 🗑️  Uninstall $APP_NAME"
        echo "5) ❌ Exit"
        echo ""
        read -p "Enter your choice (1-5): " choice
        
        case $choice in
            1)
                launch_app
                ;;
            2)
                dependency_check_menu
                ;;
            3)
                update_installation
                ;;
            4)
                uninstall_app
                ;;
            5)
                echo "Goodbye!"
                exit 0
                ;;
            *)
                log_error "Invalid choice. Please try again."
                show_main_menu
                ;;
        esac
    else
        log_info "No existing installation found."
        echo ""
        echo "Select an option:"
        echo "1) 📦 Fresh Install $APP_NAME"
        echo "2) ❌ Exit"
        echo ""
        read -p "Enter your choice (1-2): " choice
        
        case $choice in
            1)
                fresh_install
                ;;
            2)
                echo "Goodbye!"
                exit 0
                ;;
            *)
                log_error "Invalid choice. Please try again."
                show_main_menu
                ;;
        esac
    fi
}

dependency_check_menu() {
    echo ""
    echo "🔍 Dependency Check & Repair"
    echo "============================"
    
    if run_dependency_check; then
        log_success "Dependency check completed successfully!"
        echo ""
        read -p "Do you want to launch $APP_NAME now? (Y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            launch_app
        else
            show_main_menu
        fi
    else
        log_error "Dependency check failed. Some critical components are missing."
        echo ""
        echo "Options:"
        echo "1) 🔄 Try full reinstall"
        echo "2) 🔙 Back to main menu"
        echo ""
        read -p "Enter your choice (1-2): " choice
        
        case $choice in
            1)
                update_installation
                ;;
            2)
                show_main_menu
                ;;
            *)
                show_main_menu
                ;;
        esac
    fi
}

launch_app() {
    log_info "Launching $APP_NAME..."
    
    if [[ -f "$SCRIPT_DIR/AppRun.command" ]]; then
        "$SCRIPT_DIR/AppRun.command"
    else
        log_error "AppRun.command not found. Please reinstall."
        show_main_menu
    fi
}

uninstall_app() {
    echo ""
    echo "🗑️  $APP_NAME Uninstaller"
    echo "========================"
    echo ""
    echo "This will completely remove $APP_NAME and all its data."
    echo "Installation directory: $TARGET_INSTALL_DIR"
    echo ""
    
    read -p "Are you sure you want to uninstall? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Uninstalling $APP_NAME..."
        
        rm -rf "$TARGET_INSTALL_DIR"
        rm -rf "$USER_HOME/.my1ai"
        rm -f "$HOME/Desktop/🚀 Launch My1ai.command" 2>/dev/null || true
        rm -f "$SCRIPT_DIR/AppRun.command" 2>/dev/null || true
        rm -f "$SCRIPT_DIR/Uninstall.command" 2>/dev/null || true
        
        log_success "$APP_NAME has been completely uninstalled."
        echo ""
        echo "Thank you for using $APP_NAME!"
        exit 0
    else
        log_info "Uninstall cancelled."
        show_main_menu
    fi
}

update_installation() {
    echo ""
    echo "🔄 Update Installation"
    echo "====================="
    echo ""
    log_info "This will update your existing installation with the latest files and dependencies."
    echo ""
    
    read -p "Continue with update? (Y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        show_main_menu
        return
    fi
    
    # Backup existing installation
    if [[ -d "$TARGET_INSTALL_DIR" ]]; then
        log_info "Creating backup..."
        if [[ -d "$TARGET_INSTALL_DIR.backup" ]]; then
            rm -rf "$TARGET_INSTALL_DIR.backup"
        fi
        cp -r "$TARGET_INSTALL_DIR" "$TARGET_INSTALL_DIR.backup"
        log_success "Backup created at $TARGET_INSTALL_DIR.backup"
    fi
    
    # Run fresh install
    fresh_install_core
    
    # Run dependency check
    if run_dependency_check; then
        log_success "Update completed successfully!"
        echo ""
        read -p "Do you want to launch $APP_NAME now? (Y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            launch_app
        else
            show_main_menu
        fi
    else
        log_error "Update completed but dependency check failed."
        show_main_menu
    fi
}

fresh_install() {
    echo ""
    echo "📦 Fresh Install"
    echo "==============="
    echo ""
    
    fresh_install_core
    
    log_success "Fresh installation completed!"
    
    read -p "Do you want to launch $APP_NAME now? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        launch_app
    else
        show_main_menu
    fi
}

fresh_install_core() {
    # ==================================================
    # Step -1: Check System Requirements
    # ==================================================
    echo "Checking system requirements..."
    if [[ "$OSTYPE" != "darwin"* ]]; then
        log_error "This script requires macOS."
        echo "Please use the appropriate installer for your operating system."
        exit 1
    else
        log_success "macOS detected. Proceeding..."
    fi

    # Check if _internal folder exists
    if [[ ! -d "$INTERNAL_DIR" ]]; then
        log_error "_internal folder not found in $SCRIPT_DIR"
        echo "The installation package appears to be incomplete."
        echo "Please download the complete $APP_NAME package."
        exit 1
    fi

    log_success "_internal folder found with application files."
    
    # Create temporary directory
    mkdir -p "$TEMP_DIR" 2>/dev/null || true
    mkdir -p "$DOWNLOAD_DIR" 2>/dev/null || true
    mkdir -p "$USER_HOME/.my1ai" 2>/dev/null || true
    mkdir -p "$TARGET_INSTALL_DIR" 2>/dev/null || true

    # == Step 1: Copy Files from _internal Folder ==
    log_info "Copying application files..."
    
    if command -v rsync &> /dev/null; then
        rsync -av --progress "$INTERNAL_DIR/" "$TARGET_INSTALL_DIR/"
    else
        cp -r "$INTERNAL_DIR"/* "$TARGET_INSTALL_DIR/"
    fi
    
    log_success "Application files copied."

    # == Step 2: Setup Python Environment ==
    cd "$TARGET_INSTALL_DIR"
    
    # Find Python
    PYTHON_EXEC=""
    if command -v python3 &> /dev/null; then
        PYTHON_EXEC="python3"
    elif command -v python &> /dev/null; then
        PYTHON_VERSION_OUTPUT=$(python --version 2>&1)
        if [[ "$PYTHON_VERSION_OUTPUT" == *"Python 3"* ]]; then
            PYTHON_EXEC="python"
        fi
    fi

    if [[ -z "$PYTHON_EXEC" ]]; then
        log_error "Python 3 not found. Please install Python 3.10+ first."
        exit 1
    fi

    log_info "Using Python: $PYTHON_EXEC ($("$PYTHON_EXEC" --version))"

    # == Step 3: Create Virtual Environment ==
    log_info "Creating virtual environment..."
    
    if [[ -d ".venv" ]]; then
        rm -rf .venv
    fi
    
    "$PYTHON_EXEC" -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip setuptools wheel

    # == Step 4: Install Dependencies ==
    # == Step 4: Install Dependencies ==
    log_info "Installing dependencies (this may take a few minutes)..."
    
    # Phase 1: Core system packages (must succeed)
    log_info "Phase 1: Installing core system packages..."
    pip install --no-cache-dir python-dotenv PyQt6 requests markdown certifi || {
        log_error "Failed to install core packages"
        exit 1
    }
    
    # Phase 2: API providers (important for functionality)
    log_info "Phase 2: Installing API providers..."
    pip install --no-cache-dir openai groq anthropic httpx google-genai || log_warning "Some API packages failed"
    
    # Phase 3: Hugging Face ecosystem
    log_info "Phase 3: Installing Hugging Face packages..."
    pip install --no-cache-dir huggingface_hub transformers gradio_client || log_warning "Some HF packages failed"
    
    # Phase 4: Search and agent tools
    log_info "Phase 4: Installing agent tools..."
    pip install --no-cache-dir duckduckgo_search smolagents || log_warning "Some tool packages failed"
    
    # Phase 5: Document processing
    log_info "Phase 5: Installing document processing..."
    pip install --no-cache-dir pypdf2 sentence_transformers || log_warning "Some document packages failed"
    
    # Phase 6: LangChain ecosystem  
    log_info "Phase 6: Installing LangChain packages..."
    pip install --no-cache-dir langchain langchain-community langchain-huggingface || log_warning "Some LangChain packages failed"
    
    # Phase 7: Vector databases and utilities
    log_info "Phase 7: Installing vector databases..."
    pip install --no-cache-dir chromadb unidecode || log_warning "Some vector packages failed"
    
    # Phase 8: UI enhancements
    log_info "Phase 8: Installing UI packages..."
    pip install --no-cache-dir PyQt6-WebEngine || log_warning "WebEngine package failed (some features may be limited)"
    
    # Phase 9: Optional ML packages (can fail without breaking app)
    log_info "Phase 9: Installing optional ML packages..."
    pip install --no-cache-dir torch accelerate datasets || log_info "Some ML packages not installed (limited ML features)"
    
    # Phase 10: Advanced ML tools (very optional)
    log_info "Phase 10: Installing advanced ML tools..."
    pip install --no-cache-dir peft bitsandbytes || log_info "Advanced ML packages not installed"
    
    # Phase 11: Browser automation (optional)
    log_info "Phase 11: Installing browser automation..."
    pip install --no-cache-dir selenium webdriver_manager helium || log_info "Browser automation packages not installed"
    
    # Phase 12: Additional utilities
    log_info "Phase 12: Installing utilities..."
    pip install --no-cache-dir pyyaml tqdm pillow || log_warning "Some utility packages failed"
    
    log_success "Dependency installation completed."
    # == Step 5: Create Configuration ==
    if [[ ! -f ".env" ]]; then
        cat > .env << 'EOF'
# API Keys - Replace with your actual keys
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Application Settings
DEBUG=False
LOG_LEVEL=INFO
EOF
    fi

    # == Step 6: Create Launcher Scripts ==
    cat > "$SCRIPT_DIR/AppRun.command" << 'EOF'
#!/usr/bin/env bash

INSTALL_DIR="$HOME/.my1ai/desktop-app"

if [[ ! -d "$INSTALL_DIR/.venv" ]]; then
    echo "❌ My1ai is not installed properly."
    echo "Please run Install.command first."
    read -p "Press Enter to exit..."
    exit 1
fi

echo "🚀 Starting My1ai..."
cd "$INSTALL_DIR"
source .venv/bin/activate

# Quick dependency check
python -c "
try:
    from dotenv import load_dotenv
    import PyQt6.QtCore
    print('✅ Core dependencies OK')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
    print('Run Install.command to fix dependencies.')
    input('Press Enter to close...')
    exit(1)
"

if [[ $? -ne 0 ]]; then
    exit 1
fi

export PYTHONPATH="$INSTALL_DIR:$PYTHONPATH"
python main.py

if [[ $? -ne 0 ]]; then
    echo ""
    echo "❌ Application exited with an error"
    echo "Run Install.command to check dependencies."
    read -p "Press Enter to close..."
fi
EOF

    chmod +x "$SCRIPT_DIR/AppRun.command"
    
    # Create desktop shortcut
    if [[ -d "$HOME/Desktop" ]]; then
        ln -sf "$SCRIPT_DIR/AppRun.command" "$HOME/Desktop/🚀 Launch My1ai.command" 2>/dev/null || true
    fi
    
    # Cleanup
    rm -rf "$TEMP_DIR" 2>/dev/null || true
}

# ==================================================
# Main Entry Point
# ==================================================

# Check for command line arguments
if [[ $# -gt 0 ]]; then
    case $1 in
        --check-deps|--check)
            if check_existing_installation; then
                run_dependency_check
                exit $?
            else
                log_error "No installation found to check."
                exit 1
            fi
            ;;
        --install)
            fresh_install
            exit 0
            ;;
        --launch)
            if check_existing_installation; then
                launch_app
                exit 0
            else
                log_error "No installation found to launch."
                exit 1
            fi
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --check-deps    Check and repair installation"
            echo "  --install       Fresh install"
            echo "  --launch        Launch app"
            echo "  --help          Show this help"
            echo ""
            echo "Without options, shows interactive menu."
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for available options."
            exit 1
            ;;
    esac
else
    # Interactive mode
    show_main_menu
fi
#!/usr/bin/env node

/**
 * MCP Server Post-Install Script
 * 
 * This script runs after npm installation to set up the environment and install
 * Python dependencies for the MCP server.
 * 
 * It performs the following tasks:
 * 1. Checks if Python is installed
 * 2. Creates necessary directories
 * 3. Installs Python dependencies
 * 4. Sets up configuration files
 * 5. Provides next steps instructions
 */

const { spawn, execSync } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');

// Use chalk if available, otherwise define simple colored output functions
let chalk;
try {
  chalk = require('chalk');
} catch (error) {
  chalk = {
    green: (text) => `\x1b[32m${text}\x1b[0m`,
    red: (text) => `\x1b[31m${text}\x1b[0m`,
    yellow: (text) => `\x1b[33m${text}\x1b[0m`,
    blue: (text) => `\x1b[34m${text}\x1b[0m`,
    bold: (text) => `\x1b[1m${text}\x1b[0m`
  };
}

// Constants
const PYTHON_COMMANDS = ['python3', 'python', 'py'];
const MIN_PYTHON_VERSION = '3.9.0';
const REQUIRED_DIRS = [
  'app/tools',
  'app/utils',
  'app/registry',
  'tests',
  'logs',
  'output'
];

// Helper functions
function log(message, type = 'info') {
  const prefix = {
    info: chalk.blue('INFO'),
    success: chalk.green('SUCCESS'),
    warning: chalk.yellow('WARNING'),
    error: chalk.red('ERROR')
  }[type] || chalk.blue('INFO');
  
  console.log(`${prefix}: ${message}`);
}

function findPythonCommand() {
  for (const cmd of PYTHON_COMMANDS) {
    try {
      const output = execSync(`${cmd} --version`, { encoding: 'utf8' });
      const versionMatch = output.match(/Python\s+(\d+\.\d+\.\d+)/i);
      
      if (versionMatch && versionMatch[1]) {
        const version = versionMatch[1];
        
        // Check if version meets minimum requirements
        const [major, minor, patch] = version.split('.').map(Number);
        const [reqMajor, reqMinor, reqPatch] = MIN_PYTHON_VERSION.split('.').map(Number);
        
        if (
          major > reqMajor || 
          (major === reqMajor && minor > reqMinor) || 
          (major === reqMajor && minor === reqMinor && patch >= reqPatch)
        ) {
          return { command: cmd, version };
        }
        
        log(`Found Python ${version} but minimum ${MIN_PYTHON_VERSION} is required`, 'warning');
      }
    } catch (error) {
      // Command not found or execution failed, try next command
    }
  }
  
  return null;
}

function createDirectories() {
  let created = 0;
  
  for (const dir of REQUIRED_DIRS) {
    const dirPath = path.join(process.cwd(), dir);
    
    if (!fs.existsSync(dirPath)) {
      try {
        fs.mkdirSync(dirPath, { recursive: true });
        log(`Created directory: ${dir}`, 'success');
        created++;
      } catch (error) {
        log(`Failed to create directory ${dir}: ${error.message}`, 'error');
      }
    }
  }
  
  return created;
}

function createEnvFile() {
  const envPath = path.join(process.cwd(), '.env');
  
  if (!fs.existsSync(envPath)) {
    try {
      const envContent = `# MCP Server Environment Variables
SERVER_NAME="MCP Server"
SERVER_DESCRIPTION="Model Context Protocol Server for tool orchestration and agent registration"
API_PREFIX="/api/v1"
CORS_ORIGINS=["http://localhost:8000", "http://localhost:8001", "http://localhost:8080", "http://localhost:8081"]
VERSION="0.1.0"
ENVIRONMENT="development"
LOG_LEVEL="INFO"
MCP_SERVER_HOST="0.0.0.0"
MCP_SERVER_PORT=8080
`;
      
      fs.writeFileSync(envPath, envContent);
      log('Created .env file with default configuration', 'success');
      return true;
    } catch (error) {
      log(`Failed to create .env file: ${error.message}`, 'error');
      return false;
    }
  }
  
  return false;
}

function createConfigDir() {
  const configDir = path.join(os.homedir(), '.mcp');
  const configFile = path.join(configDir, 'config.json');
  
  if (!fs.existsSync(configDir)) {
    try {
      fs.mkdirSync(configDir, { recursive: true });
      
      const configContent = {
        "table_format": "markdown",
        "host": "localhost",
        "api_port": 8080,
        "ui_port": 8081,
        "mock_port": 8001,
        "env_file": ".env",
        "debug": false
      };
      
      fs.writeFileSync(configFile, JSON.stringify(configContent, null, 2));
      log('Created MCP configuration directory and config file', 'success');
      return true;
    } catch (error) {
      log(`Failed to create config directory: ${error.message}`, 'error');
      return false;
    }
  }
  
  return false;
}

function installPythonDependencies(pythonCmd) {
  return new Promise((resolve, reject) => {
    log('Installing Python dependencies...', 'info');
    
    const pip = spawn(pythonCmd, ['-m', 'pip', 'install', '-e', '.'], {
      stdio: 'pipe',
      shell: true
    });
    
    let output = '';
    
    pip.stdout.on('data', (data) => {
      output += data.toString();
      process.stdout.write('.');
    });
    
    pip.stderr.on('data', (data) => {
      output += data.toString();
      process.stderr.write('.');
    });
    
    pip.on('close', (code) => {
      console.log(''); // New line after progress dots
      
      if (code === 0) {
        log('Python dependencies installed successfully', 'success');
        resolve(true);
      } else {
        log(`Failed to install Python dependencies (exit code ${code})`, 'error');
        log(`Installation output: ${output}`, 'error');
        resolve(false);
      }
    });
    
    pip.on('error', (error) => {
      log(`Error installing Python dependencies: ${error.message}`, 'error');
      resolve(false);
    });
  });
}

function printNextSteps() {
  console.log('\n' + chalk.bold('🚀 MCP Server Installation Complete! 🚀') + '\n');
  console.log('Next steps:');
  console.log('  1. ' + chalk.yellow('Start the server:'));
  console.log('     • ' + chalk.blue('npm start') + ' - Start the MCP server');
  console.log('     • ' + chalk.blue('npm run start:all') + ' - Start both server and UI');
  console.log('     • ' + chalk.blue('npm run start:mock') + ' - Start the mock financial server');
  console.log('\n  2. ' + chalk.yellow('Use the CLI:'));
  console.log('     • ' + chalk.blue('npx mcp run') + ' - Run the server with UI');
  console.log('     • ' + chalk.blue('npx mcp status') + ' - Check server status');
  console.log('     • ' + chalk.blue('npx mcp-tools list') + ' - List available tools');
  console.log('\n  3. ' + chalk.yellow('Access the server:'));
  console.log('     • API: ' + chalk.blue('http://localhost:8080/docs'));
  console.log('     • UI: ' + chalk.blue('http://localhost:8081'));
  console.log('\nFor more information, see the documentation in the docs/ directory.');
  console.log('\n' + chalk.bold('Happy coding! 🎉') + '\n');
}

// Main function
async function main() {
  console.log(chalk.bold('\n=== MCP Server Post-Install Setup ===\n'));
  
  // Step 1: Check if Python is installed
  log('Checking for Python installation...', 'info');
  const python = findPythonCommand();
  
  if (!python) {
    log('Python 3.9+ is required but not found', 'error');
    log('Please install Python 3.9 or higher and run this script again', 'error');
    process.exit(1);
  }
  
  log(`Found Python ${python.version} (${python.command})`, 'success');
  
  // Step 2: Create directories
  log('Creating required directories...', 'info');
  const dirsCreated = createDirectories();
  
  if (dirsCreated > 0) {
    log(`Created ${dirsCreated} directories`, 'success');
  } else {
    log('All required directories already exist', 'info');
  }
  
  // Step 3: Create .env file if it doesn't exist
  log('Checking for .env file...', 'info');
  const envCreated = createEnvFile();
  
  if (!envCreated) {
    log('.env file already exists, skipping creation', 'info');
  }
  
  // Step 4: Create config directory
  log('Setting up configuration...', 'info');
  createConfigDir();
  
  // Step 5: Install Python dependencies
  const dependenciesInstalled = await installPythonDependencies(python.command);
  
  if (!dependenciesInstalled) {
    log('Python dependency installation had issues', 'warning');
    log('You may need to manually run: pip install -e .', 'warning');
  }
  
  // Step 6: Print next steps
  printNextSteps();
}

// Run the main function
main().catch(error => {
  log(`Unhandled error: ${error.message}`, 'error');
  if (error.stack) {
    console.error(error.stack);
  }
  process.exit(1);
});

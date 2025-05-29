#!/usr/bin/env node

/**
 * MCP Server CLI - Node.js Wrapper
 * 
 * This script provides a seamless interface to the Python MCP CLI for npm users.
 * It forwards commands and arguments to the Python CLI and handles errors.
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const chalk = require('chalk');
const ora = require('ora');

// Constants
const PYTHON_CMD = process.platform === 'win32' ? 'python' : 'python3';
const MCP_CLI_MODULE = 'app.cli';
const COMMANDS = [
  'run',
  'init',
  'install',
  'update',
  'status',
  'mock',
  'format-table',
  'config',
  'crm-mcp'
];

// Helper functions
function findProjectRoot() {
  // Start from current directory and go up until we find pyproject.toml
  let currentDir = process.cwd();
  
  while (currentDir !== path.parse(currentDir).root) {
    if (fs.existsSync(path.join(currentDir, 'pyproject.toml'))) {
      return currentDir;
    }
    currentDir = path.dirname(currentDir);
  }
  
  // If not found, return current directory
  return process.cwd();
}

function checkPythonInstalled() {
  try {
    const result = spawn(PYTHON_CMD, ['--version']);
    return new Promise((resolve) => {
      result.on('close', (code) => {
        resolve(code === 0);
      });
    });
  } catch (error) {
    return Promise.resolve(false);
  }
}

function checkMCPInstalled() {
  try {
    const result = spawn(PYTHON_CMD, ['-m', MCP_CLI_MODULE, '--help']);
    return new Promise((resolve) => {
      result.on('close', (code) => {
        resolve(code === 0);
      });
    });
  } catch (error) {
    return Promise.resolve(false);
  }
}

async function installMCP() {
  console.log(chalk.yellow('Installing MCP CLI...'));
  const spinner = ora('Running pip install').start();
  
  try {
    const projectRoot = findProjectRoot();
    process.chdir(projectRoot);
    
    const install = spawn(PYTHON_CMD, ['-m', 'pip', 'install', '-e', '.']);
    
    return new Promise((resolve, reject) => {
      install.on('close', (code) => {
        if (code === 0) {
          spinner.succeed('MCP CLI installed successfully');
          resolve(true);
        } else {
          spinner.fail('Failed to install MCP CLI');
          reject(new Error('Installation failed'));
        }
      });
      
      install.stderr.on('data', (data) => {
        spinner.text = `Installing: ${data.toString().trim()}`;
      });
    });
  } catch (error) {
    spinner.fail(`Installation error: ${error.message}`);
    return false;
  }
}

function showHelp() {
  console.log(chalk.bold('\nMCP Server CLI\n'));
  console.log('Usage: mcp [command] [options]\n');
  console.log('Commands:');
  console.log('  run         Run the CRM MCP Server with Chainlit UI');
  console.log('  init        Initialize the CRM MCP Server project');
  console.log('  install     Install MCP server and all dependencies');
  console.log('  update      Update MCP server to the latest version');
  console.log('  status      Check status of running MCP server instances');
  console.log('  mock        Run mock financial endpoints for testing');
  console.log('  format-table Format table data from files or stdin');
  console.log('  config      Configure CLI settings');
  console.log('  crm-mcp     Run the CRM MCP server in stdio mode');
  console.log('\nFor more information, run: mcp [command] --help');
}

// Main function
async function main() {
  // Get command and arguments
  const args = process.argv.slice(2);
  const command = args[0];
  
  // Show help if no command or help requested
  if (!command || command === '--help' || command === '-h') {
    showHelp();
    return;
  }
  
  // Check if command is valid
  if (!COMMANDS.includes(command)) {
    console.error(chalk.red(`Unknown command: ${command}`));
    console.log('Run "mcp --help" to see available commands');
    process.exit(1);
  }
  
  // Check if Python is installed
  const pythonInstalled = await checkPythonInstalled();
  if (!pythonInstalled) {
    console.error(chalk.red('Python is not installed or not in PATH'));
    console.log('Please install Python 3.9+ and try again');
    process.exit(1);
  }
  
  // Check if MCP CLI is installed
  const mcpInstalled = await checkMCPInstalled();
  if (!mcpInstalled) {
    console.log(chalk.yellow('MCP CLI not found. Attempting to install...'));
    try {
      await installMCP();
    } catch (error) {
      console.error(chalk.red(`Failed to install MCP CLI: ${error.message}`));
      console.log('Try running "pip install -e ." manually in the project root');
      process.exit(1);
    }
  }
  
  // Change to project root directory
  const projectRoot = findProjectRoot();
  process.chdir(projectRoot);
  
  // Forward command to Python CLI
  const mcpProcess = spawn(
    PYTHON_CMD, 
    ['-m', MCP_CLI_MODULE, command, ...args.slice(1)],
    { stdio: 'inherit' }
  );
  
  mcpProcess.on('close', (code) => {
    process.exit(code);
  });
  
  // Handle process termination
  process.on('SIGINT', () => {
    mcpProcess.kill('SIGINT');
  });
  
  process.on('SIGTERM', () => {
    mcpProcess.kill('SIGTERM');
  });
}

// Run the main function
main().catch(error => {
  console.error(chalk.red(`Error: ${error.message}`));
  process.exit(1);
});

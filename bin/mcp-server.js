#!/usr/bin/env node

/**
 * MCP Server Direct Launcher
 * 
 * This script launches the MCP server directly, bypassing the CLI interface.
 * It supports both HTTP and stdio modes and automatically detects the best
 * available server implementation (streamlined or original).
 * 
 * Usage:
 *   node mcp-server.js [options]
 * 
 * Options:
 *   --host <host>       Host to bind to (default: 0.0.0.0)
 *   --port <port>       Port to listen on (default: 8080)
 *   --mode <mode>       Server mode: http or stdio (default: http)
 *   --no-ui             Run without Chainlit UI
 *   --mock-port <port>  Port for mock financial server (default: 8001)
 *   --env <file>        Path to .env file (default: .env)
 *   --debug             Enable debug mode with verbose output
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const chalk = require('chalk');
const ora = require('ora');

// Parse command line arguments
const args = process.argv.slice(2);
const options = {
  host: '0.0.0.0',
  port: 8080,
  mode: 'http',
  ui: true,
  mockPort: 8001,
  env: '.env',
  debug: false
};

// Parse arguments
for (let i = 0; i < args.length; i++) {
  const arg = args[i];
  
  if (arg === '--host' && i + 1 < args.length) {
    options.host = args[++i];
  } else if (arg === '--port' && i + 1 < args.length) {
    options.port = parseInt(args[++i], 10);
  } else if (arg === '--mode' && i + 1 < args.length) {
    options.mode = args[++i];
    if (!['http', 'stdio'].includes(options.mode)) {
      console.error(chalk.red(`Invalid mode: ${options.mode}. Must be 'http' or 'stdio'`));
      process.exit(1);
    }
  } else if (arg === '--no-ui') {
    options.ui = false;
  } else if (arg === '--mock-port' && i + 1 < args.length) {
    options.mockPort = parseInt(args[++i], 10);
  } else if (arg === '--env' && i + 1 < args.length) {
    options.env = args[++i];
  } else if (arg === '--debug') {
    options.debug = true;
  } else if (arg === '--help' || arg === '-h') {
    console.log(chalk.bold('\nMCP Server Direct Launcher\n'));
    console.log('Usage: mcp-server [options]\n');
    console.log('Options:');
    console.log('  --host <host>       Host to bind to (default: 0.0.0.0)');
    console.log('  --port <port>       Port to listen on (default: 8080)');
    console.log('  --mode <mode>       Server mode: http or stdio (default: http)');
    console.log('  --no-ui             Run without Chainlit UI');
    console.log('  --mock-port <port>  Port for mock financial server (default: 8001)');
    console.log('  --env <file>        Path to .env file (default: .env)');
    console.log('  --debug             Enable debug mode with verbose output');
    console.log('  --help, -h          Show this help message');
    process.exit(0);
  }
}

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
  const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
  
  try {
    const result = spawn(pythonCmd, ['--version']);
    return new Promise((resolve) => {
      result.on('close', (code) => {
        resolve(code === 0 ? pythonCmd : null);
      });
    });
  } catch (error) {
    return Promise.resolve(null);
  }
}

function checkServerModule(pythonCmd, modulePath) {
  try {
    const result = spawn(pythonCmd, ['-c', `import ${modulePath}`]);
    return new Promise((resolve) => {
      result.on('close', (code) => {
        resolve(code === 0);
      });
    });
  } catch (error) {
    return Promise.resolve(false);
  }
}

async function startServer() {
  const projectRoot = findProjectRoot();
  process.chdir(projectRoot);
  
  console.log(chalk.blue('Starting MCP Server...'));
  console.log(chalk.dim(`Project root: ${projectRoot}`));
  
  // Check Python installation
  const pythonCmd = await checkPythonInstalled();
  if (!pythonCmd) {
    console.error(chalk.red('Python is not installed or not in PATH'));
    console.log('Please install Python 3.9+ and try again');
    process.exit(1);
  }
  
  // Check for .env file
  if (options.env && fs.existsSync(options.env)) {
    console.log(chalk.green(`✓ Using environment file: ${options.env}`));
    // Load environment variables
    require('dotenv').config({ path: options.env });
  } else {
    console.log(chalk.yellow(`Warning: Environment file ${options.env} not found`));
  }
  
  // Determine which server module to use
  const hasStreamlinedServer = await checkServerModule(pythonCmd, 'app.streamlined_mcp_server');
  const serverModule = hasStreamlinedServer ? 'app.streamlined_mcp_server' : 'app.mcp_server';
  
  console.log(chalk.green(`✓ Using server module: ${serverModule}`));
  
  // Start the server based on mode
  if (options.mode === 'stdio') {
    // Start in stdio mode
    console.log(chalk.blue('Starting MCP server in stdio mode...'));
    
    const serverProcess = spawn(
      pythonCmd,
      ['-m', serverModule, '--mode', 'stdio'],
      { stdio: 'inherit' }
    );
    
    // Handle process events
    serverProcess.on('close', (code) => {
      if (code !== 0) {
        console.error(chalk.red(`MCP server exited with code ${code}`));
      }
      process.exit(code);
    });
    
    // Handle process termination
    process.on('SIGINT', () => {
      console.log(chalk.yellow('\nShutting down MCP server...'));
      serverProcess.kill('SIGINT');
    });
    
    process.on('SIGTERM', () => {
      serverProcess.kill('SIGTERM');
    });
  } else {
    // Start in HTTP mode
    console.log(chalk.blue(`Starting MCP server in HTTP mode on ${options.host}:${options.port}...`));
    
    // Start the API server
    let serverCmd;
    if (hasStreamlinedServer) {
      serverCmd = [
        '-m', 'uvicorn',
        'app.streamlined_mcp_server:app',
        '--host', options.host,
        '--port', options.port.toString(),
        '--reload'
      ];
    } else {
      serverCmd = [
        '-m', 'uvicorn',
        'app.mcp_server:main_factory',
        '--host', options.host,
        '--port', options.port.toString(),
        '--reload',
        '--factory'
      ];
    }
    
    if (options.debug) {
      serverCmd.push('--log-level', 'debug');
    }
    
    const serverProcess = spawn(
      pythonCmd,
      serverCmd,
      { stdio: 'pipe' }
    );
    
    // Capture and display server output
    serverProcess.stdout.on('data', (data) => {
      process.stdout.write(`[API] ${data}`);
    });
    
    serverProcess.stderr.on('data', (data) => {
      process.stderr.write(`[API] ${data}`);
    });
    
    // Start Chainlit UI if enabled
    let uiProcess = null;
    if (options.ui) {
      console.log(chalk.blue(`Starting Chainlit UI on ${options.host}:${options.port + 1}...`));
      
      uiProcess = spawn(
        pythonCmd,
        [
          '-m', 'chainlit',
          'run', 'ui/app.py',
          '--host', options.host,
          '--port', (options.port + 1).toString()
        ],
        { stdio: 'pipe' }
      );
      
      // Capture and display UI output
      uiProcess.stdout.on('data', (data) => {
        process.stdout.write(`[UI] ${data}`);
      });
      
      uiProcess.stderr.on('data', (data) => {
        process.stderr.write(`[UI] ${data}`);
      });
      
      // Handle UI process events
      uiProcess.on('close', (code) => {
        if (code !== 0 && code !== null) {
          console.error(chalk.red(`Chainlit UI exited with code ${code}`));
        }
        
        // Kill the server process if UI exits
        if (serverProcess.exitCode === null) {
          serverProcess.kill();
        }
      });
    }
    
    // Handle server process events
    serverProcess.on('close', (code) => {
      if (code !== 0 && code !== null) {
        console.error(chalk.red(`MCP server exited with code ${code}`));
      }
      
      // Kill the UI process if server exits
      if (uiProcess && uiProcess.exitCode === null) {
        uiProcess.kill();
      }
      
      process.exit(code || 0);
    });
    
    // Handle process termination
    process.on('SIGINT', () => {
      console.log(chalk.yellow('\nShutting down servers...'));
      
      if (uiProcess) {
        uiProcess.kill('SIGINT');
      }
      
      serverProcess.kill('SIGINT');
    });
    
    process.on('SIGTERM', () => {
      if (uiProcess) {
        uiProcess.kill('SIGTERM');
      }
      
      serverProcess.kill('SIGTERM');
    });
    
    // Print URLs
    setTimeout(() => {
      console.log(chalk.green('\n✨ Server URLs:'));
      console.log(chalk.green(`API: http://${options.host === '0.0.0.0' ? 'localhost' : options.host}:${options.port}`));
      
      if (options.ui) {
        console.log(chalk.green(`UI: http://${options.host === '0.0.0.0' ? 'localhost' : options.host}:${options.port + 1}`));
      }
      
      console.log(chalk.yellow('\nPress Ctrl+C to stop the server(s)'));
    }, 2000);
  }
}

// Start the server
startServer().catch(error => {
  console.error(chalk.red(`Error starting MCP server: ${error.message}`));
  if (options.debug) {
    console.error(error.stack);
  }
  process.exit(1);
});

#!/usr/bin/env node

/**
 * MCP Tools Utility
 * 
 * A utility script for inspecting and testing MCP tools directly.
 * This tool allows you to list, inspect, and execute MCP tools without
 * going through the natural language processing pipeline.
 * 
 * Features:
 * - List all available tools with descriptions
 * - Inspect tool schemas and parameters
 * - Execute tools directly with parameters
 * - Format tool outputs as tables (markdown, HTML, CSV, JSON)
 * - Test tools with mock data
 * 
 * Usage:
 *   mcp-tools list                     List all available tools
 *   mcp-tools inspect <tool>           Inspect a specific tool's schema
 *   mcp-tools execute <tool> [params]  Execute a tool with parameters
 *   mcp-tools format <file> [options]  Format a data file as table
 *   mcp-tools mock <tool>              Generate mock data for a tool
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const chalk = require('chalk');
const ora = require('ora');
const axios = require('axios');
const { table, getBorderCharacters } = require('table');
const inquirer = require('inquirer');
const { program } = require('commander');

// Constants
const PYTHON_CMD = process.platform === 'win32' ? 'python' : 'python3';
const DEFAULT_HOST = 'localhost';
const DEFAULT_PORT = 8080;
const TABLE_FORMATS = ['markdown', 'html', 'csv', 'json', 'table'];

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

async function checkServerRunning(host = DEFAULT_HOST, port = DEFAULT_PORT) {
  try {
    const response = await axios.get(`http://${host}:${port}/ping`, { timeout: 2000 });
    return response.status === 200;
  } catch (error) {
    return false;
  }
}

async function startServerIfNeeded(host = DEFAULT_HOST, port = DEFAULT_PORT) {
  const serverRunning = await checkServerRunning(host, port);
  
  if (!serverRunning) {
    console.log(chalk.yellow('MCP Server is not running. Starting it now...'));
    
    const spinner = ora('Starting MCP Server').start();
    
    // Start server in background
    const projectRoot = findProjectRoot();
    const serverProcess = spawn(
      PYTHON_CMD,
      ['-m', 'uvicorn', 'app.streamlined_mcp_server:app', '--host', host, '--port', port.toString()],
      { 
        cwd: projectRoot,
        detached: true,
        stdio: 'ignore'
      }
    );
    
    // Detach the process so it runs independently
    serverProcess.unref();
    
    // Wait for server to start
    let attempts = 0;
    const maxAttempts = 10;
    
    while (attempts < maxAttempts) {
      await new Promise(resolve => setTimeout(resolve, 1000));
      const running = await checkServerRunning(host, port);
      
      if (running) {
        spinner.succeed('MCP Server started successfully');
        return true;
      }
      
      attempts++;
      spinner.text = `Starting MCP Server (attempt ${attempts}/${maxAttempts})`;
    }
    
    spinner.fail('Failed to start MCP Server');
    return false;
  }
  
  return true;
}

async function fetchTools(host = DEFAULT_HOST, port = DEFAULT_PORT) {
  try {
    const response = await axios.get(`http://${host}:${port}/tools`);
    return response.data.tools || [];
  } catch (error) {
    console.error(chalk.red(`Error fetching tools: ${error.message}`));
    return [];
  }
}

async function executeTool(toolName, params = {}, host = DEFAULT_HOST, port = DEFAULT_PORT) {
  try {
    const response = await axios.post(`http://${host}:${port}/execute`, {
      tool_name: toolName,
      parameters: params,
      context: {}
    });
    
    return response.data;
  } catch (error) {
    console.error(chalk.red(`Error executing tool: ${error.message}`));
    if (error.response && error.response.data) {
      console.error(chalk.red(`Server response: ${JSON.stringify(error.response.data)}`));
    }
    return { error: error.message };
  }
}

function formatAsTable(data, format = 'table') {
  if (!data || (Array.isArray(data) && data.length === 0)) {
    return 'No data available';
  }
  
  // Handle string responses
  if (typeof data === 'string') {
    return data;
  }
  
  // Handle error responses
  if (data.error) {
    return chalk.red(`Error: ${data.error}`);
  }
  
  // Handle result from execute endpoint
  if (data.result !== undefined) {
    if (typeof data.result === 'string') {
      return data.result;
    }
    data = data.result;
  }
  
  // If data is not an array but an object, convert to array with one item
  if (!Array.isArray(data) && typeof data === 'object') {
    data = [data];
  }
  
  // If data is still not an array, return as string
  if (!Array.isArray(data)) {
    return String(data);
  }
  
  // Get all unique keys from all objects
  const keys = Array.from(new Set(
    data.flatMap(item => Object.keys(item))
  ));
  
  if (keys.length === 0) {
    return 'Empty data set';
  }
  
  // Format based on requested format
  if (format === 'json') {
    return JSON.stringify(data, null, 2);
  } else if (format === 'csv') {
    const header = keys.join(',');
    const rows = data.map(item => 
      keys.map(key => 
        item[key] !== undefined ? String(item[key]).replace(/,/g, ';') : ''
      ).join(',')
    );
    return [header, ...rows].join('\n');
  } else if (format === 'html') {
    const header = `<tr>${keys.map(key => `<th>${key}</th>`).join('')}</tr>`;
    const rows = data.map(item => 
      `<tr>${keys.map(key => `<td>${item[key] !== undefined ? item[key] : ''}</td>`).join('')}</tr>`
    );
    return `<table border="1">\n<thead>${header}</thead>\n<tbody>${rows.join('\n')}</tbody>\n</table>`;
  } else if (format === 'markdown') {
    const header = `| ${keys.join(' | ')} |`;
    const separator = `| ${keys.map(() => '---').join(' | ')} |`;
    const rows = data.map(item => 
      `| ${keys.map(key => item[key] !== undefined ? item[key] : '').join(' | ')} |`
    );
    return [header, separator, ...rows].join('\n');
  } else {
    // Default to console table
    const tableData = [
      keys.map(key => chalk.bold(key)),
      ...data.map(item => keys.map(key => item[key] !== undefined ? item[key] : ''))
    ];
    
    return table(tableData, {
      border: getBorderCharacters('norc')
    });
  }
}

function generateMockData(toolSchema) {
  const result = {};
  
  if (!toolSchema || !toolSchema.parameters || !toolSchema.parameters.properties) {
    return result;
  }
  
  const properties = toolSchema.parameters.properties;
  
  for (const [key, prop] of Object.entries(properties)) {
    // Skip context parameter which is usually handled by the server
    if (key === 'ctx' || key === 'context') {
      continue;
    }
    
    const type = prop.type || 'string';
    
    // Generate appropriate mock value based on type
    switch (type) {
      case 'string':
        if (key.toLowerCase().includes('date')) {
          result[key] = '2025-05-29';
        } else if (key.toLowerCase().includes('email')) {
          result[key] = 'user@example.com';
        } else if (key.toLowerCase().includes('name')) {
          result[key] = 'Test User';
        } else if (key.toLowerCase().includes('region')) {
          result[key] = 'USA';
        } else if (key.toLowerCase().includes('currency')) {
          result[key] = 'USD';
        } else {
          result[key] = `test_${key}`;
        }
        break;
      case 'integer':
      case 'number':
        if (key.toLowerCase().includes('limit')) {
          result[key] = 10;
        } else if (key.toLowerCase().includes('id')) {
          result[key] = 12345;
        } else {
          result[key] = 42;
        }
        break;
      case 'boolean':
        result[key] = true;
        break;
      case 'array':
        result[key] = [];
        break;
      case 'object':
        result[key] = {};
        break;
      default:
        result[key] = null;
    }
  }
  
  return result;
}

async function saveOutputToFile(output, filename) {
  try {
    fs.writeFileSync(filename, output);
    console.log(chalk.green(`✓ Output saved to ${filename}`));
  } catch (error) {
    console.error(chalk.red(`Error saving output: ${error.message}`));
  }
}

// Define the CLI program
program
  .name('mcp-tools')
  .description('Utility for inspecting and testing MCP tools')
  .version('0.1.0');

// List command
program
  .command('list')
  .description('List all available tools')
  .option('-h, --host <host>', 'Server host', DEFAULT_HOST)
  .option('-p, --port <port>', 'Server port', DEFAULT_PORT)
  .option('-f, --format <format>', 'Output format (table, json, markdown)', 'table')
  .option('-o, --output <file>', 'Save output to file')
  .action(async (options) => {
    const spinner = ora('Fetching tools').start();
    
    try {
      // Start server if needed
      const serverRunning = await startServerIfNeeded(options.host, options.port);
      if (!serverRunning) {
        spinner.fail('Could not start MCP Server');
        process.exit(1);
      }
      
      // Fetch tools
      const tools = await fetchTools(options.host, options.port);
      spinner.succeed(`Found ${tools.length} tools`);
      
      // Format tools for display
      const toolsData = tools.map(tool => ({
        name: tool.name,
        description: tool.description,
        namespace: tool.namespace || 'default',
        parameters: Object.keys(tool.parameters?.properties || {}).join(', ')
      }));
      
      // Display tools
      const output = formatAsTable(toolsData, options.format);
      console.log(output);
      
      // Save to file if requested
      if (options.output) {
        await saveOutputToFile(output, options.output);
      }
    } catch (error) {
      spinner.fail(`Error: ${error.message}`);
      process.exit(1);
    }
  });

// Inspect command
program
  .command('inspect <tool>')
  .description('Inspect a specific tool\'s schema')
  .option('-h, --host <host>', 'Server host', DEFAULT_HOST)
  .option('-p, --port <port>', 'Server port', DEFAULT_PORT)
  .option('-f, --format <format>', 'Output format (json, markdown)', 'json')
  .option('-o, --output <file>', 'Save output to file')
  .action(async (toolName, options) => {
    const spinner = ora(`Inspecting tool: ${toolName}`).start();
    
    try {
      // Start server if needed
      const serverRunning = await startServerIfNeeded(options.host, options.port);
      if (!serverRunning) {
        spinner.fail('Could not start MCP Server');
        process.exit(1);
      }
      
      // Fetch tools
      const tools = await fetchTools(options.host, options.port);
      const tool = tools.find(t => t.name === toolName);
      
      if (!tool) {
        spinner.fail(`Tool not found: ${toolName}`);
        process.exit(1);
      }
      
      spinner.succeed(`Found tool: ${toolName}`);
      
      // Display tool schema
      let output;
      if (options.format === 'json') {
        output = JSON.stringify(tool, null, 2);
        console.log(output);
      } else if (options.format === 'markdown') {
        output = `# Tool: ${tool.name}\n\n`;
        output += `**Description**: ${tool.description}\n\n`;
        output += `**Namespace**: ${tool.namespace || 'default'}\n\n`;
        
        if (tool.parameters && tool.parameters.properties) {
          output += '## Parameters\n\n';
          
          for (const [paramName, paramSchema] of Object.entries(tool.parameters.properties)) {
            output += `### ${paramName}\n\n`;
            output += `- **Type**: ${paramSchema.type || 'any'}\n`;
            if (paramSchema.description) {
              output += `- **Description**: ${paramSchema.description}\n`;
            }
            if (paramSchema.default !== undefined) {
              output += `- **Default**: ${paramSchema.default}\n`;
            }
            output += '\n';
          }
        }
        
        console.log(output);
      } else {
        output = JSON.stringify(tool, null, 2);
        console.log(output);
      }
      
      // Save to file if requested
      if (options.output) {
        await saveOutputToFile(output, options.output);
      }
    } catch (error) {
      spinner.fail(`Error: ${error.message}`);
      process.exit(1);
    }
  });

// Execute command
program
  .command('execute <tool>')
  .description('Execute a tool with parameters')
  .option('-h, --host <host>', 'Server host', DEFAULT_HOST)
  .option('-p, --port <port>', 'Server port', DEFAULT_PORT)
  .option('-f, --format <format>', 'Output format (table, json, markdown, html, csv)', 'table')
  .option('-o, --output <file>', 'Save output to file')
  .option('-i, --interactive', 'Interactive parameter input mode')
  .option('-p, --params <json>', 'Parameters as JSON string')
  .option('-m, --mock', 'Use mock data for parameters')
  .action(async (toolName, options) => {
    let spinner = ora(`Executing tool: ${toolName}`).start();
    
    try {
      // Start server if needed
      const serverRunning = await startServerIfNeeded(options.host, options.port);
      if (!serverRunning) {
        spinner.fail('Could not start MCP Server');
        process.exit(1);
      }
      
      // Fetch tools
      const tools = await fetchTools(options.host, options.port);
      const tool = tools.find(t => t.name === toolName);
      
      if (!tool) {
        spinner.fail(`Tool not found: ${toolName}`);
        process.exit(1);
      }
      
      spinner.succeed(`Found tool: ${toolName}`);
      
      // Get parameters
      let params = {};
      
      if (options.mock) {
        // Generate mock data
        params = generateMockData(tool);
        console.log(chalk.yellow('Using mock parameters:'));
        console.log(JSON.stringify(params, null, 2));
      } else if (options.params) {
        // Parse parameters from command line
        try {
          params = JSON.parse(options.params);
        } catch (error) {
          console.error(chalk.red(`Error parsing parameters: ${error.message}`));
          process.exit(1);
        }
      } else if (options.interactive) {
        // Interactive parameter input
        spinner.stop();
        
        if (tool.parameters && tool.parameters.properties) {
          const questions = [];
          
          for (const [paramName, paramSchema] of Object.entries(tool.parameters.properties)) {
            // Skip context parameter
            if (paramName === 'ctx' || paramName === 'context') {
              continue;
            }
            
            const question = {
              name: paramName,
              message: `Enter ${paramName}${paramSchema.description ? ` (${paramSchema.description})` : ''}:`,
              default: paramSchema.default
            };
            
            // Set appropriate question type based on parameter type
            if (paramSchema.type === 'boolean') {
              question.type = 'confirm';
            } else if (paramSchema.enum) {
              question.type = 'list';
              question.choices = paramSchema.enum;
            } else {
              question.type = 'input';
            }
            
            questions.push(question);
          }
          
          if (questions.length > 0) {
            console.log(chalk.blue(`\nEnter parameters for ${toolName}:`));
            params = await inquirer.prompt(questions);
          } else {
            console.log(chalk.yellow('No parameters required for this tool.'));
          }
        } else {
          console.log(chalk.yellow('No parameters schema available for this tool.'));
        }
        
        spinner = ora(`Executing tool: ${toolName}`).start();
      }
      
      // Execute the tool
      spinner.text = `Executing ${toolName} with parameters...`;
      const result = await executeTool(toolName, params, options.host, options.port);
      
      if (result.error) {
        spinner.fail(`Error executing tool: ${result.error}`);
        process.exit(1);
      }
      
      spinner.succeed(`Tool ${toolName} executed successfully`);
      
      // Format and display result
      const output = formatAsTable(result, options.format);
      console.log('\nResult:');
      console.log(output);
      
      // Save to file if requested
      if (options.output) {
        await saveOutputToFile(output, options.output);
      }
    } catch (error) {
      spinner.fail(`Error: ${error.message}`);
      process.exit(1);
    }
  });

// Format command
program
  .command('format <file>')
  .description('Format a data file as table')
  .option('-f, --format <format>', 'Output format (markdown, html, csv, json, table)', 'table')
  .option('-o, --output <file>', 'Save output to file')
  .action(async (file, options) => {
    const spinner = ora(`Formatting file: ${file}`).start();
    
    try {
      // Check if file exists
      if (!fs.existsSync(file)) {
        spinner.fail(`File not found: ${file}`);
        process.exit(1);
      }
      
      // Read file content
      const content = fs.readFileSync(file, 'utf8');
      let data;
      
      // Parse file based on extension
      const ext = path.extname(file).toLowerCase();
      
      if (ext === '.json') {
        data = JSON.parse(content);
        
        // Handle common API response format
        if (data.data) {
          data = data.data;
        } else if (data.result) {
          data = data.result;
        }
      } else if (ext === '.csv') {
        // Simple CSV parsing
        const lines = content.split('\n').filter(line => line.trim());
        const headers = lines[0].split(',').map(h => h.trim());
        
        data = lines.slice(1).map(line => {
          const values = line.split(',').map(v => v.trim());
          const obj = {};
          
          headers.forEach((header, index) => {
            obj[header] = values[index] || '';
          });
          
          return obj;
        });
      } else {
        spinner.fail(`Unsupported file format: ${ext}`);
        process.exit(1);
      }
      
      spinner.succeed(`File loaded: ${file}`);
      
      // Format and display data
      const output = formatAsTable(data, options.format);
      console.log(output);
      
      // Save to file if requested
      if (options.output) {
        await saveOutputToFile(output, options.output);
      }
    } catch (error) {
      spinner.fail(`Error: ${error.message}`);
      process.exit(1);
    }
  });

// Mock command
program
  .command('mock <tool>')
  .description('Generate mock data for a tool')
  .option('-h, --host <host>', 'Server host', DEFAULT_HOST)
  .option('-p, --port <port>', 'Server port', DEFAULT_PORT)
  .option('-f, --format <format>', 'Output format (json, markdown)', 'json')
  .option('-o, --output <file>', 'Save output to file')
  .option('-e, --execute', 'Execute the tool with mock data')
  .action(async (toolName, options) => {
    const spinner = ora(`Generating mock data for tool: ${toolName}`).start();
    
    try {
      // Start server if needed
      const serverRunning = await startServerIfNeeded(options.host, options.port);
      if (!serverRunning) {
        spinner.fail('Could not start MCP Server');
        process.exit(1);
      }
      
      // Fetch tools
      const tools = await fetchTools(options.host, options.port);
      const tool = tools.find(t => t.name === toolName);
      
      if (!tool) {
        spinner.fail(`Tool not found: ${toolName}`);
        process.exit(1);
      }
      
      // Generate mock data
      const mockData = generateMockData(tool);
      spinner.succeed(`Generated mock data for tool: ${toolName}`);
      
      // Format and display mock data
      let output;
      if (options.format === 'json') {
        output = JSON.stringify(mockData, null, 2);
      } else if (options.format === 'markdown') {
        output = `# Mock Data for ${toolName}\n\n`;
        output += '```json\n';
        output += JSON.stringify(mockData, null, 2);
        output += '\n```\n';
      } else {
        output = JSON.stringify(mockData, null, 2);
      }
      
      console.log(output);
      
      // Save to file if requested
      if (options.output) {
        await saveOutputToFile(output, options.output);
      }
      
      // Execute with mock data if requested
      if (options.execute) {
        console.log(chalk.blue('\nExecuting tool with mock data...'));
        
        const result = await executeTool(toolName, mockData, options.host, options.port);
        
        if (result.error) {
          console.error(chalk.red(`Error executing tool: ${result.error}`));
        } else {
          console.log(chalk.green('Tool executed successfully'));
          console.log('\nResult:');
          console.log(formatAsTable(result, 'table'));
        }
      }
    } catch (error) {
      spinner.fail(`Error: ${error.message}`);
      process.exit(1);
    }
  });

// Parse arguments and execute command
program.parse(process.argv);

// Show help if no command provided
if (!process.argv.slice(2).length) {
  program.outputHelp();
}

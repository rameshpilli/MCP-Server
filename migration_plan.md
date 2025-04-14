# Migration Plan: Moving to MCP Package Database

## Overview

This document outlines the steps to transition from our custom database implementation to using the MCP package's built-in database functionality.

## Files to Remove

These files will be completely removed as their functionality will be provided by the MCP package:

1. `app/core/database.py` - Replace with MCP database functionality
2. `app/core/models.py` - Replace with MCP models
3. `app/core/migrations/` - Remove entire directory, will use MCP migrations
4. `app/core/dependencies.py` - Database dependencies replaced by MCP

## Files to Modify

1. `app/api/mcp_server.py`
   - Update imports to use MCP models
   - Replace custom DB accesses with MCP DB functions

2. `mcp_server.py`
   - Replace database initialization with MCP initialization
   - Update server lifespan function

3. `app/core/config.py`
   - Simplify database configuration to use MCP settings
   - Remove redundant settings

4. `app/core/storage.py` 
   - Keep useful storage backends as needed
   - Ensure compatibility with MCP interfaces

5. `app/main.py`
   - Update to use MCP database initialization
   - Update API router dependencies

## New Files to Create

1. `app/mcp_models.py` - Create adapters between our model schema and MCP models if needed
2. `app/config/mcp_config.py` - MCP-specific configuration

## Implementation Steps

1. Install the latest MCP package with database support
2. Analyze MCP database schema and models
3. Create migration scripts to adapt existing data to MCP format
4. Update imports across codebase
5. Update resource and tool handlers to use MCP data access
6. Test each component individually
7. Remove redundant files
8. Comprehensive integration testing

## Rollback Plan

1. Create database dumps before migration
2. Keep backup of all removed files
3. Create version branch before migration
4. Document all changes for potential rollback 
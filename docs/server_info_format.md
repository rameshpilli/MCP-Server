# `server_info` Output Format

The `server_info` tool returns Markdown describing the server configuration and all registered tools. Tools are grouped by namespace and show their input parameters as bullet lists.

## Example Output

```markdown
# MCP Server Information
**Host**: localhost
**Port**: 8080

\U0001F4E6 Available MCP Tools

### finance
\U0001F527 `validate_company`
Description: Validate if a company exists and return standardized info
Inputs:
- `company_id`: string – Ticker or company identifier

\U0001F527 `validate_currency`
Description: Validate currency code and optionally convert amount
Inputs:
- `amount`: number – Amount to validate
- `currency`: string – Currency code
- `target_currency`: string – Target currency for conversion

### document
\U0001F527 `summarize_doc`
Description: Summarize a specific document by name
Inputs:
- `doc_name`: string – Name of the document
```

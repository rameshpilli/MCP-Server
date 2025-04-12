# MCP (Model Control Platform)

A flexible and enterprise-ready system for managing and monitoring multiple LLM models in a corporate environment.

## Features

- 🚀 Multi-model support with unified API
- 📊 Comprehensive monitoring and observability
- 🔐 Enterprise-grade security
- 🔄 Automatic failover and load balancing
- 📝 Detailed logging and analytics
- ☁️ Azure integration (optional)
- 🗄️ Multiple database backend support

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MCP.git
cd MCP
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Initialize database:
```bash
alembic upgrade head
```

6. Start the service:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Documentation

Detailed documentation is available in the `docs/` directory:

- [Setup and Configuration](docs/setup_and_configuration.md)
- [API Reference](docs/api_reference.md)
- [Monitoring Guide](docs/monitoring.md)
- [Azure Integration](docs/azure_integration.md)

## Architecture

```
app/
├── core/           # Core functionality
│   ├── config.py   # Configuration management
│   ├── database.py # Database operations
│   └── monitoring.py # Monitoring system
├── models/         # Model definitions
├── api/           # API endpoints
└── utils/         # Utility functions
```

## Configuration

The system can be configured using environment variables or configuration files. Key configuration options:

```bash
# Storage Backend
STORAGE_BACKEND=local  # or 'azure'

# Database
AZURE_DB_URL=your-connection-string  # For Azure SQL

# Monitoring
APPINSIGHTS_CONNECTION_STRING=your-connection-string
MAX_ERROR_RATE=0.05
MAX_LATENCY_MS=1000
```

See [Configuration Guide](docs/setup_and_configuration.md) for more details.

## Development

1. Create a new branch:
```bash
git checkout -b feature/your-feature
```

2. Make your changes and run tests:
```bash
pytest tests/
```

3. Submit a pull request

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support and questions:
- Create an issue
- Contact the development team
- Check the documentation

## Acknowledgments

- Built with FastAPI and SQLAlchemy
- Monitoring powered by OpenTelemetry
- Azure integration using official Azure SDKs 
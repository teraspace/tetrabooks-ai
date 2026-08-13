# ADR-001: Pragmatic clean architecture

## Context
The system must swap model and retrieval vendors without changing use cases.

## Decision
Use small provider ports and keep FastAPI endpoints thin. Domain objects do not import SQLAlchemy or vendor SDKs.

## Alternatives
A direct SDK call in each route would be shorter initially but would make Azure/OpenAI changes invasive.

## Consequences
More explicit wiring, but easier testing and interview discussion. We deliberately avoid empty abstractions.

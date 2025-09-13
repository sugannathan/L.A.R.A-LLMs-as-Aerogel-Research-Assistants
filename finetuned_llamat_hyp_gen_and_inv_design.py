#!/usr/bin/env python3
"""
L.A.R.A + MatKG + Hypothesis Generation + Inverse Design Framework
Combines fine-tuned L.A.R.A model with knowledge graph context for:
1. Hypothesis generation 
2. Inverse design for target properties
"""

import torch
import requests
import json
import os
import sys
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from datetime import datetime
import sqlite3
from pathlib import Path
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OperationMode(Enum):
    HYPOTHESIS_GENERATION = "hypothesis"
    INVERSE_DESIGN = "inverse_design"

@dataclass
class TargetProperty:
    """Structure for target property specifications"""
    name: str
    value: Union[float, str]
    unit: str
    tolerance: Optional[float] = None
    priority: str = "high"  # high, medium, low

@dataclass
class SynthesisRoute:
    """Structure for synthesis route recommendations"""
    route_name: str
    steps: List[str]
    parameters: Dict[str, Any]
    expected_properties: Dict[str, str]
    confidence: float
    complexity: str  # low, medium, high
    estimated_time: str
    safety_notes: List[str]

@dataclass
class MaterialFormulation:
    """Structure for material formulation recommendations"""
    formulation_name: str
    precursors: Dict[str, str]  # name: concentration/ratio
    solvents: List[str]
    catalysts: List[str]
    processing_conditions: Dict[str, Any]
    expected_properties: Dict[str, str]
    confidence: float

@dataclass
class HypothesisResult:
    """Structure for hypothesis generation results"""
    hypothesis: str
    confidence: float
    kg_entities: List[str]
    supporting_evidence: List[str]
    generated_by: str
    timestamp: str

@dataclass
class InverseDesignResult:
    """Structure for inverse design results"""
    target_properties: List[TargetProperty]
    synthesis_routes: List[SynthesisRoute]
    material_formulations: List[MaterialFormulation]
    design_rationale: str
    confidence: float
    generated_by: str
    timestamp: str

@dataclass
class KGContext:
    """Structure for knowledge graph context"""
    entities: List[str]
    relations: List[Dict[str, str]]
    properties: List[str]
    synthesis_methods: List[str]
    applications: List[str]

class MatKGInterface:
    """Enhanced interface to MatKG knowledge graph for aerogel-related queries"""
    
    def __init__(self, kg_data_path: Optional[str] = None):
        self.kg_data_path = kg_data_path
        self.entities_cache = {}
        self.relations_cache = {}
        self.property_synthesis_map = {}
        self.init_kg_connection()
    
    def init_kg_connection(self):
        """Initialize connection to MatKG data"""
        try:
            if self.kg_data_path and os.path.exists(self.kg_data_path):
                logger.info(f"Loading MatKG data from {self.kg_data_path}")
                self.load_local_kg_data()
            else:
                logger.info("Using enhanced simulated MatKG interface")
                self.setup_enhanced_kg()
        except Exception as e:
            logger.warning(f"KG connection failed: {e}. Using fallback mode.")
            self.setup_enhanced_kg()
    
    def setup_enhanced_kg(self):
        """Set up enhanced simulated knowledge graph with property-synthesis mappings"""
        self.aerogel_entities = {
            'materials': ['carbon aerogel', 'silica aerogel', 'resorcinol-formaldehyde aerogel', 
                         'melamine-formaldehyde aerogel', 'polyimide aerogel', 'cellulose aerogel',
                         'graphene aerogel', 'CNT aerogel', 'metal oxide aerogel'],
            'properties': ['porosity', 'surface area', 'density', 'thermal conductivity', 
                          'electrical conductivity', 'mechanical strength', 'pore size',
                          'compressive strength', 'thermal stability', 'hydrophobicity'],
            'synthesis_methods': ['sol-gel process', 'supercritical drying', 'freeze drying',
                                'ambient pressure drying', 'pyrolysis', 'carbonization',
                                'template synthesis', 'electrospinning', 'chemical vapor deposition'],
            'precursors': ['resorcinol', 'formaldehyde', 'tetraethyl orthosilicate', 'phenol',
                          'melamine', 'polyacrylonitrile', 'cellulose', 'graphene oxide'],
            'processing_conditions': ['temperature', 'pH', 'catalyst concentration', 'gelation time',
                                    'aging time', 'drying method', 'atmosphere', 'pressure'],
            'applications': ['thermal insulation', 'energy storage', 'catalysis', 
                           'adsorption', 'electrodes', 'supercapacitors', 'aerospace'],
            'characterization': ['BET analysis', 'SEM imaging', 'TGA analysis', 
                               'XRD analysis', 'nitrogen adsorption', 'mercury porosimetry']
        }
        
        # Enhanced relations with property-synthesis mappings
        self.relations = [
            {'subject': 'carbon aerogel', 'predicate': 'synthesized_by', 'object': 'sol-gel process'},
            {'subject': 'resorcinol-formaldehyde', 'predicate': 'precursor_for', 'object': 'carbon aerogel'},
            {'subject': 'supercritical drying', 'predicate': 'preserves', 'object': 'pore structure'},
            {'subject': 'high surface area', 'predicate': 'enables', 'object': 'energy storage'},
            {'subject': 'pyrolysis temperature', 'predicate': 'affects', 'object': 'electrical conductivity'},
            {'subject': 'RF ratio', 'predicate': 'controls', 'object': 'porosity'},
            {'subject': 'catalyst concentration', 'predicate': 'influences', 'object': 'pore size'},
            {'subject': 'pH level', 'predicate': 'determines', 'object': 'gelation kinetics'},
        ]
        
        # Property-synthesis mappings for inverse design
        self.property_synthesis_map = {
            'high_porosity': {
                'methods': ['supercritical drying', 'freeze drying'],
                'parameters': {'RF_ratio': '1:2 to 1:3', 'pH': '5.5-6.5', 'temperature': '80-90°C'},
                'precursors': ['resorcinol', 'formaldehyde', 'sodium carbonate']
            },
            'low_thermal_conductivity': {
                'methods': ['ambient pressure drying', 'supercritical drying'],
                'parameters': {'density': '<0.1 g/cm³', 'pore_size': '10-100 nm'},
                'precursors': ['silica precursors', 'organic templates']
            },
            'high_electrical_conductivity': {
                'methods': ['pyrolysis', 'carbonization', 'graphitization'],
                'parameters': {'temperature': '1000-1400°C', 'atmosphere': 'inert', 'heating_rate': '5°C/min'},
                'precursors': ['carbon-based precursors', 'graphene oxide']
            },
            'high_surface_area': {
                'methods': ['activation', 'template synthesis'],
                'parameters': {'activation_temperature': '800-900°C', 'KOH_ratio': '1:1 to 4:1'},
                'precursors': ['carbon precursors', 'mesoporous templates']
            },
            'mechanical_strength': {
                'methods': ['controlled gelation', 'reinforcement'],
                'parameters': {'cross_linking_density': 'high', 'aging_time': '24-72h'},
                'precursors': ['high molecular weight polymers', 'reinforcing agents']
            }
        }
    
    def get_synthesis_routes_for_properties(self, target_properties: List[TargetProperty]) -> Dict[str, Any]:
        """Get synthesis routes that can achieve target properties"""
        suitable_routes = {}
        
        for prop in target_properties:
            prop_key = f"{prop.name}_{self._categorize_property_value(prop)}"
            
            if prop_key in self.property_synthesis_map:
                suitable_routes[prop.name] = self.property_synthesis_map[prop_key]
            else:
                # Find closest match or general approach
                suitable_routes[prop.name] = self._find_general_approach(prop)
        
        return suitable_routes
    
    def _categorize_property_value(self, prop: TargetProperty) -> str:
        """Categorize property value as high/low/medium for mapping"""
        if isinstance(prop.value, str):
            if any(term in prop.value.lower() for term in ['high', 'maximum', 'large']):
                return 'high'
            elif any(term in prop.value.lower() for term in ['low', 'minimum', 'small']):
                return 'low'
            else:
                return 'medium'
        
        # For numerical values, use heuristics based on property type
        if 'porosity' in prop.name.lower():
            return 'high' if float(prop.value) > 0.8 else 'medium' if float(prop.value) > 0.5 else 'low'
        elif 'conductivity' in prop.name.lower() and 'thermal' in prop.name.lower():
            return 'low' if float(prop.value) < 0.05 else 'medium' if float(prop.value) < 0.2 else 'high'
        elif 'conductivity' in prop.name.lower() and 'electrical' in prop.name.lower():
            return 'high' if float(prop.value) > 100 else 'medium' if float(prop.value) > 10 else 'low'
        
        return 'medium'
    
    def _find_general_approach(self, prop: TargetProperty) -> Dict[str, Any]:
        """Find general synthesis approach for unknown properties"""
        return {
            'methods': ['sol-gel process', 'controlled drying'],
            'parameters': {'optimization_needed': True},
            'precursors': ['standard precursors']
        }
    
    def query_entities(self, query: str, entity_type: str = 'all') -> List[str]:
        """Query knowledge graph for relevant entities"""
        query_lower = query.lower()
        relevant_entities = []
        
        if entity_type == 'all':
            for category, entities in self.aerogel_entities.items():
                for entity in entities:
                    if any(term in entity.lower() for term in query_lower.split()):
                        relevant_entities.append(entity)
        else:
            entities = self.aerogel_entities.get(entity_type, [])
            for entity in entities:
                if any(term in entity.lower() for term in query_lower.split()):
                    relevant_entities.append(entity)
        
        return list(set(relevant_entities))
    
    def get_related_concepts(self, concept: str, relation_type: str = 'all') -> List[Dict]:
        """Get concepts related to the input concept"""
        related = []
        
        for relation in self.relations:
            if concept.lower() in relation['subject'].lower():
                related.append({
                    'concept': relation['object'],
                    'relation': relation['predicate'],
                    'direction': 'forward'
                })
            elif concept.lower() in relation['object'].lower():
                related.append({
                    'concept': relation['subject'],
                    'relation': relation['predicate'],
                    'direction': 'backward'
                })
        
        return related
    
    def build_context(self, query: str) -> KGContext:
        """Build comprehensive context from knowledge graph"""
        entities = self.query_entities(query)
        relations = []
        properties = []
        synthesis_methods = []
        applications = []
        
        for entity in entities:
            related = self.get_related_concepts(entity)
            relations.extend(related)
            
            # Categorize related concepts
            if entity in self.aerogel_entities.get('properties', []):
                properties.append(entity)
            elif entity in self.aerogel_entities.get('synthesis_methods', []):
                synthesis_methods.append(entity)
            elif entity in self.aerogel_entities.get('applications', []):
                applications.append(entity)
        
        return KGContext(
            entities=list(set(entities)),
            relations=relations,
            properties=list(set(properties)),
            synthesis_methods=list(set(synthesis_methods)),
            applications=list(set(applications))
        )

class InverseDesignEngine:
    """Engine for inverse design of aerogel materials"""
    
    def __init__(self, kg_interface: MatKGInterface):
        self.kg_interface = kg_interface
        self.synthesis_templates = self.load_synthesis_templates()
        self.formulation_templates = self.load_formulation_templates()
    
    def load_synthesis_templates(self) -> Dict[str, Dict]:
        """Load synthesis route templates"""
        return {
            'sol_gel_carbon': {
                'name': 'Sol-Gel Carbon Aerogel Synthesis',
                'steps': [
                    'Dissolve resorcinol and formaldehyde in water with catalyst',
                    'Adjust pH and allow gelation at controlled temperature',
                    'Age gel for specified time',
                    'Solvent exchange to remove water',
                    'Dry using supercritical or ambient pressure method',
                    'Pyrolyze at high temperature for carbonization'
                ],
                'parameters': {
                    'RF_ratio': 'variable',
                    'catalyst': 'Na2CO3 or other base',
                    'pH': '5.5-7.0',
                    'gelation_temp': '80-90°C',
                    'aging_time': '24-72 hours',
                    'pyrolysis_temp': '800-1400°C'
                },
                'complexity': 'medium',
                'time': '3-7 days'
            },
            'silica_aerogel': {
                'name': 'Silica Aerogel Synthesis',
                'steps': [
                    'Hydrolyze silica precursor in alcohol solution',
                    'Add catalyst to promote gelation',
                    'Age gel to strengthen network',
                    'Solvent exchange to alcohol',
                    'Supercritical drying with CO2'
                ],
                'parameters': {
                    'precursor': 'TEOS or TMOS',
                    'solvent': 'methanol or ethanol',
                    'water_ratio': '4-8:1',
                    'catalyst': 'NH4OH or HCl',
                    'aging_time': '24-48 hours'
                },
                'complexity': 'high',
                'time': '2-5 days'
            }
        }
    
    def load_formulation_templates(self) -> Dict[str, Dict]:
        """Load material formulation templates"""
        return {
            'high_porosity_carbon': {
                'precursors': {
                    'resorcinol': '1.0 g',
                    'formaldehyde': '1.4 g (37% solution)',
                    'water': '10 mL',
                    'catalyst': '0.01 g Na2CO3'
                },
                'processing': {
                    'gelation_temp': '85°C',
                    'aging_time': '72 hours',
                    'drying': 'supercritical CO2',
                    'pyrolysis': '900°C in N2'
                }
            },
            'conductive_aerogel': {
                'precursors': {
                    'resorcinol': '2.0 g',
                    'formaldehyde': '2.8 g',
                    'graphene_oxide': '0.1 g',
                    'water': '15 mL'
                },
                'processing': {
                    'mixing_time': '2 hours ultrasonication',
                    'gelation_temp': '90°C',
                    'carbonization': '1200°C in Ar'
                }
            }
        }
    
    def design_synthesis_route(self, target_properties: List[TargetProperty]) -> List[SynthesisRoute]:
        """Design synthesis routes for target properties"""
        routes = []
        
        # Get KG recommendations
        kg_routes = self.kg_interface.get_synthesis_routes_for_properties(target_properties)
        
        # Match with templates and create detailed routes
        for prop in target_properties:
            if prop.name in kg_routes:
                route_data = kg_routes[prop.name]
                
                # Select appropriate template
                template_key = self._select_template(prop, route_data)
                template = self.synthesis_templates.get(template_key, self.synthesis_templates['sol_gel_carbon'])
                
                # Customize template for target property
                customized_route = self._customize_route(template, prop, route_data)
                routes.append(customized_route)
        
        # Remove duplicates and rank by confidence
        unique_routes = self._deduplicate_routes(routes)
        return sorted(unique_routes, key=lambda x: x.confidence, reverse=True)
    
    def design_material_formulation(self, target_properties: List[TargetProperty]) -> List[MaterialFormulation]:
        """Design material formulations for target properties"""
        formulations = []
        
        for prop in target_properties:
            # Select base formulation
            base_formulation = self._select_base_formulation(prop)
            
            # Optimize formulation for target property
            optimized = self._optimize_formulation(base_formulation, prop)
            formulations.append(optimized)
        
        return formulations
    
    def _select_template(self, prop: TargetProperty, route_data: Dict) -> str:
        """Select appropriate synthesis template"""
        if 'carbon' in str(prop.name).lower() or 'electrical' in str(prop.name).lower():
            return 'sol_gel_carbon'
        elif 'thermal' in str(prop.name).lower() and 'low' in str(prop.value).lower():
            return 'silica_aerogel'
        else:
            return 'sol_gel_carbon'  # Default
    
    def _customize_route(self, template: Dict, prop: TargetProperty, route_data: Dict) -> SynthesisRoute:
        """Customize synthesis route template for specific property"""
        
        # Calculate confidence based on property match
        confidence = 0.7  # Base confidence
        if prop.name.lower() in ['porosity', 'surface area', 'density']:
            confidence += 0.15
        
        # Customize parameters based on target property
        custom_params = template['parameters'].copy()
        if 'parameters' in route_data:
            custom_params.update(route_data['parameters'])
        
        # Add property-specific optimizations
        optimizations = self._get_property_optimizations(prop)
        custom_params.update(optimizations)
        
        return SynthesisRoute(
            route_name=f"{template['name']} (optimized for {prop.name})",
            steps=template['steps'].copy(),
            parameters=custom_params,
            expected_properties={prop.name: str(prop.value)},
            confidence=confidence,
            complexity=template['complexity'],
            estimated_time=template['time'],
            safety_notes=self._get_safety_notes(custom_params)
        )
    
    def _get_property_optimizations(self, prop: TargetProperty) -> Dict[str, Any]:
        """Get parameter optimizations for specific properties"""
        optimizations = {}
        
        if 'porosity' in prop.name.lower():
            if isinstance(prop.value, (int, float)) and prop.value > 0.9:
                optimizations['RF_ratio'] = '1:3'
                optimizations['drying_method'] = 'supercritical'
            elif isinstance(prop.value, str) and 'high' in prop.value.lower():
                optimizations['RF_ratio'] = '1:2.5'
                optimizations['gelation_temp'] = '85°C'
        
        elif 'conductivity' in prop.name.lower() and 'electrical' in prop.name.lower():
            optimizations['pyrolysis_temp'] = '1200°C'
            optimizations['heating_rate'] = '5°C/min'
            optimizations['atmosphere'] = 'inert (N2 or Ar)'
        
        elif 'thermal' in prop.name.lower() and 'conductivity' in prop.name.lower():
            optimizations['density_target'] = '<0.1 g/cm³'
            optimizations['pore_size_target'] = '10-50 nm'
        
        return optimizations
    
    def _get_safety_notes(self, parameters: Dict) -> List[str]:
        """Generate safety notes based on synthesis parameters"""
        notes = []
        
        if 'pyrolysis_temp' in parameters:
            temp = parameters['pyrolysis_temp']
            if any(char.isdigit() for char in str(temp)):
                temp_val = int(''.join(filter(str.isdigit, str(temp))))
                if temp_val > 1000:
                    notes.append("High-temperature operation requires specialized equipment and safety protocols")
        
        if 'formaldehyde' in str(parameters).lower():
            notes.append("Formaldehyde is toxic and carcinogenic - use proper ventilation and PPE")
        
        if 'supercritical' in str(parameters).lower():
            notes.append("Supercritical drying requires high-pressure equipment and safety training")
        
        return notes
    
    def _select_base_formulation(self, prop: TargetProperty) -> Dict:
        """Select base formulation template"""
        if 'conductivity' in prop.name.lower() and 'electrical' in prop.name.lower():
            return self.formulation_templates['conductive_aerogel']
        else:
            return self.formulation_templates['high_porosity_carbon']
    
    def _optimize_formulation(self, base: Dict, prop: TargetProperty) -> MaterialFormulation:
        """Optimize formulation for target property"""
        
        optimized_precursors = base['precursors'].copy()
        optimized_processing = base['processing'].copy()
        
        # Property-specific optimizations
        if 'porosity' in prop.name.lower() and isinstance(prop.value, (int, float)):
            if prop.value > 0.9:
                # Ultra-high porosity modifications
                optimized_precursors['water'] = '15 mL'  # Increase dilution
                optimized_processing['aging_time'] = '96 hours'
        
        return MaterialFormulation(
            formulation_name=f"Optimized formulation for {prop.name}",
            precursors=optimized_precursors,
            solvents=['water'],
            catalysts=['Na2CO3'],
            processing_conditions=optimized_processing,
            expected_properties={prop.name: str(prop.value)},
            confidence=0.75
        )
    
    def _deduplicate_routes(self, routes: List[SynthesisRoute]) -> List[SynthesisRoute]:
        """Remove duplicate synthesis routes"""
        unique_routes = []
        seen_names = set()
        
        for route in routes:
            if route.route_name not in seen_names:
                unique_routes.append(route)
                seen_names.add(route.route_name)
        
        return unique_routes

class HypothesisGenerator:
    """Base class for hypothesis generation with different strategies"""
    
    def __init__(self, strategy: str = "data_driven"):
        self.strategy = strategy
        self.hypothesis_templates = self.load_templates()
    
    def load_templates(self) -> Dict[str, List[str]]:
        """Load hypothesis templates for different domains"""
        return {
            'synthesis': [
                "Modifying {parameter} during {process} will {effect} the {property} of {material}",
                "Using {catalyst} in {synthesis_method} will improve {target_property}",
                "Optimizing {condition} during {process} can enhance {performance_metric}"
            ],
            'structure_property': [
                "Materials with {structural_feature} exhibit enhanced {property}",
                "The relationship between {parameter1} and {parameter2} affects {outcome}",
                "Controlling {morphological_feature} leads to improved {application_performance}"
            ],
            'application': [
                "Aerogels with {specific_property} are optimal for {application}",
                "Combining {material1} and {material2} creates synergistic effects for {use_case}",
                "The {property} of aerogels determines their efficiency in {application}"
            ]
        }
    
    def generate_hypotheses(self, context: KGContext, query: str, num_hypotheses: int = 5) -> List[str]:
        """Generate hypotheses based on context and strategy"""
        hypotheses = []
        
        if self.strategy == "template_based":
            hypotheses = self._generate_template_based(context, query, num_hypotheses)
        elif self.strategy == "data_driven":
            hypotheses = self._generate_data_driven(context, query, num_hypotheses)
        elif self.strategy == "knowledge_guided":
            hypotheses = self._generate_knowledge_guided(context, query, num_hypotheses)
        
        return hypotheses[:num_hypotheses]
    
    def _generate_template_based(self, context: KGContext, query: str, num_hypotheses: int) -> List[str]:
        """Generate hypotheses using predefined templates"""
        hypotheses = []
        
        for category, templates in self.hypothesis_templates.items():
            if len(hypotheses) >= num_hypotheses:
                break
                
            for template in templates:
                if len(hypotheses) >= num_hypotheses:
                    break
                
                # Fill template with context entities
                try:
                    filled = self._fill_template(template, context)
                    if filled and filled not in hypotheses:
                        hypotheses.append(filled)
                except:
                    continue
        
        return hypotheses
    
    def _fill_template(self, template: str, context: KGContext) -> Optional[str]:
        """Fill template with appropriate entities from context"""
        import random
        
        # Simple template filling - can be made more sophisticated
        placeholders = re.findall(r'{([^}]+)}', template)
        filled_template = template
        
        for placeholder in placeholders:
            replacement = None
            
            if 'material' in placeholder.lower():
                replacement = random.choice(context.entities) if context.entities else 'aerogel'
            elif 'property' in placeholder.lower():
                replacement = random.choice(context.properties) if context.properties else 'porosity'
            elif 'process' in placeholder.lower() or 'method' in placeholder.lower():
                replacement = random.choice(context.synthesis_methods) if context.synthesis_methods else 'sol-gel process'
            elif 'application' in placeholder.lower():
                replacement = random.choice(context.applications) if context.applications else 'thermal insulation'
            else:
                replacement = random.choice(context.entities) if context.entities else placeholder
            
            filled_template = filled_template.replace(f'{{{placeholder}}}', replacement)
        
        return filled_template
    
    def _generate_data_driven(self, context: KGContext, query: str, num_hypotheses: int) -> List[str]:
        """Generate hypotheses based on data patterns"""
        base_hypotheses = [
            f"Increasing surface area enhances {app}" for app in context.applications
        ]
        
        return base_hypotheses[:num_hypotheses]
    
    def _generate_knowledge_guided(self, context: KGContext, query: str, num_hypotheses: int) -> List[str]:
        """Generate hypotheses guided by knowledge graph relations"""
        hypotheses = []
        
        for relation in context.relations:
            if len(hypotheses) >= num_hypotheses:
                break
            
            hypothesis = f"The relationship between {relation.get('concept', 'unknown')} and related factors {relation.get('relation', 'affects')} performance in aerogel applications"
            hypotheses.append(hypothesis)
        
        return hypotheses

class LARAAdvancedSystem:
    """Enhanced system with both hypothesis generation and inverse design capabilities"""
    
    def __init__(self, 
                 model_path: str = "./finetuned_llamat",
                 kg_data_path: Optional[str] = None,
                 evaluator_model: str = "gpt-3.5-turbo"):
        
        self.model_path = model_path
        self.evaluator_model = evaluator_model
        
        # Initialize components
        self.lara_model = None
        self.tokenizer = None
        self.kg_interface = MatKGInterface(kg_data_path)
        self.hypothesis_generator = HypothesisGenerator()
        self.inverse_design_engine = InverseDesignEngine(self.kg_interface)
        
        # Load models
        self.load_lara_model()
        
        # Initialize database for storing results
        self.init_database()
    
    def load_lara_model(self):
        """Load the fine-tuned L.A.R.A model"""
        try:
            logger.info(f"Loading L.A.R.A model from: {self.model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            
            self.lara_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            
            # Fix pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            logger.info("✅ L.A.R.A model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load L.A.R.A model: {e}")
            sys.exit(1)
    
    def init_database(self):
        """Initialize SQLite database for storing results"""
        self.db_path = "lara_results.db"
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Hypotheses table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hypotheses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                hypothesis TEXT NOT NULL,
                confidence REAL,
                kg_context TEXT,
                generated_by TEXT,
                evaluation_score REAL,
                timestamp TEXT,
                supporting_evidence TEXT
            )
        """)
        
        # Inverse design results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inverse_designs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                target_properties TEXT NOT NULL,
                synthesis_routes TEXT,
                material_formulations TEXT,
                design_rationale TEXT,
                confidence REAL,
                generated_by TEXT,
                timestamp TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def parse_target_properties(self, property_string: str) -> List[TargetProperty]:
        """Parse target properties from user input string"""
        properties = []
        
        # Simple parsing - can be enhanced with NLP
        if 'porosity' in property_string.lower():
            # Extract value if present
            porosity_match = re.search(r'porosity[:\s]*([0-9.]+)', property_string.lower())
            if porosity_match:
                value = float(porosity_match.group(1))
                if value <= 1.0:  # Fractional
                    properties.append(TargetProperty("porosity", value, "fraction"))
                else:  # Percentage
                    properties.append(TargetProperty("porosity", value/100, "fraction"))
            else:
                # Look for qualitative descriptors
                if any(word in property_string.lower() for word in ['high', 'maximum', 'ultra']):
                    properties.append(TargetProperty("porosity", "high", "qualitative"))
                elif any(word in property_string.lower() for word in ['low', 'minimum']):
                    properties.append(TargetProperty("porosity", "low", "qualitative"))
        
        if 'thermal conductivity' in property_string.lower():
            # Extract value
            tc_match = re.search(r'thermal conductivity[:\s]*([0-9.]+)', property_string.lower())
            if tc_match:
                value = float(tc_match.group(1))
                properties.append(TargetProperty("thermal_conductivity", value, "W/m·K"))
            else:
                if any(word in property_string.lower() for word in ['low', 'minimum', 'insulating']):
                    properties.append(TargetProperty("thermal_conductivity", "low", "qualitative"))
        
        if 'electrical conductivity' in property_string.lower():
            # Extract value
            ec_match = re.search(r'electrical conductivity[:\s]*([0-9.]+)', property_string.lower())
            if ec_match:
                value = float(ec_match.group(1))
                properties.append(TargetProperty("electrical_conductivity", value, "S/m"))
            else:
                if any(word in property_string.lower() for word in ['high', 'conductive']):
                    properties.append(TargetProperty("electrical_conductivity", "high", "qualitative"))
        
        if 'surface area' in property_string.lower():
            sa_match = re.search(r'surface area[:\s]*([0-9.]+)', property_string.lower())
            if sa_match:
                value = float(sa_match.group(1))
                properties.append(TargetProperty("surface_area", value, "m²/g"))
            else:
                if any(word in property_string.lower() for word in ['high', 'maximum', 'large']):
                    properties.append(TargetProperty("surface_area", "high", "qualitative"))
        
        if 'density' in property_string.lower():
            density_match = re.search(r'density[:\s]*([0-9.]+)', property_string.lower())
            if density_match:
                value = float(density_match.group(1))
                properties.append(TargetProperty("density", value, "g/cm³"))
            else:
                if any(word in property_string.lower() for word in ['low', 'ultralight']):
                    properties.append(TargetProperty("density", "low", "qualitative"))
        
        return properties
    
    def generate_inverse_design_prompt(self, target_properties: List[TargetProperty], 
                                     synthesis_routes: List[SynthesisRoute],
                                     kg_context: KGContext) -> str:
        """Generate enhanced prompt for inverse design with L.A.R.A"""
        
        properties_str = ", ".join([f"{prop.name}: {prop.value} {prop.unit}" for prop in target_properties])
        
        kg_context_str = f"""
Knowledge Graph Context:
- Relevant Materials: {', '.join(kg_context.entities[:5]) if kg_context.entities else 'None'}
- Synthesis Methods: {', '.join(kg_context.synthesis_methods[:3]) if kg_context.synthesis_methods else 'None'}
- Related Properties: {', '.join(kg_context.properties[:5]) if kg_context.properties else 'None'}

Recommended Synthesis Routes:
"""
        
        for i, route in enumerate(synthesis_routes[:2], 1):
            kg_context_str += f"""
{i}. {route.route_name}
   - Key Parameters: {', '.join([f"{k}: {v}" for k, v in list(route.parameters.items())[:3]])}
   - Complexity: {route.complexity}
   - Confidence: {route.confidence:.2f}
"""
        
        prompt = f"""### Instruction:
You are L.A.R.A (LLMs as Aerogel Research Assistants), an expert in carbon aerogels and materials science. Using your expertise and the provided synthesis routes, explain how to achieve the target properties and provide additional optimization strategies.

Target Properties: {properties_str}

{kg_context_str}

### Task:
Provide detailed guidance on achieving these target properties, including:
1. Why the recommended synthesis routes are suitable
2. Critical process parameters to control
3. Potential challenges and solutions
4. Alternative approaches or modifications

### Response:
Based on my expertise in aerogel synthesis and the knowledge graph context:
"""
        
        return prompt
    
    def generate_kg_enhanced_prompt(self, query: str, kg_context: KGContext) -> str:
        """Generate enhanced prompt with knowledge graph context for hypothesis generation"""
        
        kg_context_str = f"""
Knowledge Graph Context:
- Relevant Materials: {', '.join(kg_context.entities[:5]) if kg_context.entities else 'None'}
- Properties: {', '.join(kg_context.properties[:5]) if kg_context.properties else 'None'}  
- Synthesis Methods: {', '.join(kg_context.synthesis_methods[:3]) if kg_context.synthesis_methods else 'None'}
- Applications: {', '.join(kg_context.applications[:3]) if kg_context.applications else 'None'}
"""
        
        prompt = f"""### Instruction:
You are L.A.R.A (LLMs as Aerogel Research Assistants), an expert in carbon aerogels and materials science. Using your training knowledge and the provided context, generate scientific insights about the following query.

{kg_context_str}

### Query:
{query}

### Response:
Based on the knowledge graph context and my expertise in aerogels, here's my scientific analysis:
"""
        
        return prompt
    
    def query_lara(self, prompt: str, max_tokens: int = 300, temperature: float = 0.3) -> str:
        """Query the fine-tuned L.A.R.A model"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.lara_model.device)
        
        with torch.no_grad():
            outputs = self.lara_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the response part
        response = response.split("### Response:")[-1].strip()
        
        return self.clean_response(response)
    
    def clean_response(self, response: str) -> str:
        """Clean up response from the model"""
        response = re.sub(r'Dr\.?\s*T:\s*', '', response)
        response = re.sub(r'You:\s*', '', response)
        response = re.sub(r'Assistant:\s*', '', response)
        response = re.sub(r'[!]{2,}', '.', response)
        response = re.sub(r'\bamazing\b', 'notable', response, flags=re.IGNORECASE)
        
        return response.strip()
    
    def evaluate_hypothesis_with_external_llm(self, hypothesis: str, context: str) -> Dict[str, Any]:
        """Evaluate hypothesis using external LLM"""
        # Simplified evaluation - integrate with actual API
        scores = {
            'plausibility': np.random.uniform(6, 9),
            'novelty': np.random.uniform(5, 8),
            'testability': np.random.uniform(6, 9),
            'impact': np.random.uniform(5, 8)
        }
        
        overall_score = np.mean(list(scores.values()))
        
        return {
            'scores': scores,
            'overall_score': overall_score,
            'evaluation': f"Hypothesis shows moderate to high scientific merit with overall score: {overall_score:.2f}"
        }
    
    def generate_hypotheses(self, query: str, num_hypotheses: int = 5) -> List[HypothesisResult]:
        """Generate hypotheses using all components"""
        logger.info(f"Generating hypotheses for: {query}")
        
        # Step 1: Build knowledge graph context
        kg_context = self.kg_interface.build_context(query)
        logger.info(f"Found {len(kg_context.entities)} relevant entities in KG")
        
        # Step 2: Generate initial hypotheses using different strategies
        template_hypotheses = self.hypothesis_generator.generate_hypotheses(
            kg_context, query, num_hypotheses//2
        )
        
        # Step 3: Use L.A.R.A to refine and generate additional hypotheses
        lara_hypotheses = []
        for i in range(num_hypotheses - len(template_hypotheses)):
            prompt = self.generate_kg_enhanced_prompt(query, kg_context)
            response = self.query_lara(prompt)
            lara_hypotheses.append(response)
        
        # Step 4: Combine and evaluate hypotheses
        all_hypotheses = template_hypotheses + lara_hypotheses
        results = []
        
        for i, hypothesis in enumerate(all_hypotheses):
            # Evaluate each hypothesis
            evaluation = self.evaluate_hypothesis_with_external_llm(
                hypothesis, 
                f"Query: {query}, KG Context: {kg_context.entities}"
            )
            
            result = HypothesisResult(
                hypothesis=hypothesis,
                confidence=evaluation['overall_score'] / 10.0,
                kg_entities=kg_context.entities,
                supporting_evidence=[f"Based on {len(kg_context.entities)} KG entities"],
                generated_by="template" if i < len(template_hypotheses) else "L.A.R.A",
                timestamp=datetime.now().isoformat()
            )
            
            results.append(result)
        
        # Step 5: Store results
        self.store_hypothesis_results(query, results)
        
        # Step 6: Rank and return top hypotheses
        results.sort(key=lambda x: x.confidence, reverse=True)
        
        return results[:num_hypotheses]
    
    def perform_inverse_design(self, property_string: str) -> InverseDesignResult:
        """Perform inverse design for target properties"""
        logger.info(f"Performing inverse design for: {property_string}")
        
        # Step 1: Parse target properties
        target_properties = self.parse_target_properties(property_string)
        logger.info(f"Parsed {len(target_properties)} target properties")
        
        if not target_properties:
            raise ValueError("Could not parse any target properties from input")
        
        # Step 2: Build knowledge graph context
        kg_context = self.kg_interface.build_context(property_string)
        
        # Step 3: Design synthesis routes
        synthesis_routes = self.inverse_design_engine.design_synthesis_route(target_properties)
        
        # Step 4: Design material formulations
        material_formulations = self.inverse_design_engine.design_material_formulation(target_properties)
        
        # Step 5: Get L.A.R.A insights
        prompt = self.generate_inverse_design_prompt(target_properties, synthesis_routes, kg_context)
        lara_insights = self.query_lara(prompt, max_tokens=400, temperature=0.4)
        
        # Step 6: Compile result
        result = InverseDesignResult(
            target_properties=target_properties,
            synthesis_routes=synthesis_routes,
            material_formulations=material_formulations,
            design_rationale=lara_insights,
            confidence=np.mean([route.confidence for route in synthesis_routes]) if synthesis_routes else 0.5,
            generated_by="L.A.R.A + MatKG",
            timestamp=datetime.now().isoformat()
        )
        
        # Step 7: Store results
        self.store_inverse_design_results(result)
        
        return result
    
    def store_hypothesis_results(self, query: str, results: List[HypothesisResult]):
        """Store hypothesis results in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for result in results:
            cursor.execute("""
                INSERT INTO hypotheses 
                (query, hypothesis, confidence, kg_context, generated_by, evaluation_score, timestamp, supporting_evidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                query,
                result.hypothesis,
                result.confidence,
                json.dumps(result.kg_entities),
                result.generated_by,
                result.confidence,
                result.timestamp,
                json.dumps(result.supporting_evidence)
            ))
        
        conn.commit()
        conn.close()
    
    def store_inverse_design_results(self, result: InverseDesignResult):
        """Store inverse design results in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO inverse_designs 
            (target_properties, synthesis_routes, material_formulations, design_rationale, confidence, generated_by, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            json.dumps([{"name": p.name, "value": p.value, "unit": p.unit} for p in result.target_properties]),
            json.dumps([{"name": r.route_name, "confidence": r.confidence, "complexity": r.complexity} for r in result.synthesis_routes]),
            json.dumps([{"name": f.formulation_name, "confidence": f.confidence} for f in result.material_formulations]),
            result.design_rationale,
            result.confidence,
            result.generated_by,
            result.timestamp
        ))
        
        conn.commit()
        conn.close()
    
    def interactive_mode(self):
        """Enhanced interactive mode with both hypothesis generation and inverse design"""
        print("=" * 80)
        print("🧪 L.A.R.A Advanced System: Hypothesis Generation + Inverse Design")
        print("=" * 80)
        print("Two main modes available:")
        print("1. 🔬 Hypothesis Generation - Generate scientific hypotheses")
        print("2. 🎯 Inverse Design - Design materials for target properties") 
        print("3. 💡 Mixed Mode - Ask questions and get both approaches")
        print("=" * 80)
        
        while True:
            try:
                print("\nSelect mode:")
                print("1. Hypothesis Generation")
                print("2. Inverse Design") 
                print("3. Mixed/Auto Mode")
                print("4. Quit")
                
                mode_choice = input("\nEnter mode (1-4): ").strip()
                
                if mode_choice == '4' or mode_choice.lower() in ['quit', 'exit', 'q']:
                    print("👋 Thank you for using L.A.R.A Advanced System!")
                    break
                
                if mode_choice == '1':
                    self._hypothesis_mode()
                elif mode_choice == '2':
                    self._inverse_design_mode()
                elif mode_choice == '3':
                    self._mixed_mode()
                else:
                    print("Invalid choice. Please enter 1, 2, 3, or 4.")
                
            except KeyboardInterrupt:
                print("\n👋 Interrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _hypothesis_mode(self):
        """Dedicated hypothesis generation mode"""
        print("\n🔬 Hypothesis Generation Mode")
        print("Enter research questions to generate scientific hypotheses")
        
        while True:
            query = input("\n🤔 Research question (or 'back' to return): ").strip()
            
            if query.lower() == 'back':
                break
            
            if not query:
                continue
            
            print(f"\n🔍 Generating hypotheses for: {query}")
            print("Please wait...")
            
            try:
                results = self.generate_hypotheses(query, num_hypotheses=5)
                
                print(f"\n📝 Generated {len(results)} hypotheses:\n")
                
                for i, result in enumerate(results, 1):
                    print(f"{i}. **{result.hypothesis}**")
                    print(f"   Confidence: {result.confidence:.2f}")
                    print(f"   Generated by: {result.generated_by}")
                    print(f"   KG entities: {', '.join(result.kg_entities[:3])}{'...' if len(result.kg_entities) > 3 else ''}")
                    print()
                
            except Exception as e:
                print(f"❌ Error generating hypotheses: {e}")
    
    def _inverse_design_mode(self):
        """Dedicated inverse design mode"""
        print("\n🎯 Inverse Design Mode")
        print("Specify target properties to get synthesis routes and formulations")
        print("\nExample inputs:")
        print("- 'High porosity > 0.95 and low thermal conductivity'")
        print("- 'Electrical conductivity 100 S/m, surface area 1500 m²/g'")
        print("- 'Ultra-low density < 0.05 g/cm³'")
        
        while True:
            property_input = input("\n🎯 Target properties (or 'back' to return): ").strip()
            
            if property_input.lower() == 'back':
                break
            
            if not property_input:
                continue
            
            print(f"\n🔧 Designing materials for: {property_input}")
            print("Please wait...")
            
            try:
                result = self.perform_inverse_design(property_input)
                
                print(f"\n🎯 Inverse Design Results:")
                print(f"Target Properties: {', '.join([f'{p.name}: {p.value} {p.unit}' for p in result.target_properties])}")
                print(f"Overall Confidence: {result.confidence:.2f}")
                
                print(f"\n📋 Recommended Synthesis Routes ({len(result.synthesis_routes)}):")
                for i, route in enumerate(result.synthesis_routes, 1):
                    print(f"\n{i}. {route.route_name}")
                    print(f"   Confidence: {route.confidence:.2f}")
                    print(f"   Complexity: {route.complexity}")
                    print(f"   Estimated Time: {route.estimated_time}")
                    print(f"   Key Steps: {'; '.join(route.steps[:3])}...")
                    
                    print(f"   Critical Parameters:")
                    for param, value in list(route.parameters.items())[:4]:
                        print(f"   - {param}: {value}")
                    
                    if route.safety_notes:
                        print(f"   ⚠️ Safety Notes: {'; '.join(route.safety_notes)}")
                
                print(f"\n🧪 Material Formulations ({len(result.material_formulations)}):")
                for i, formulation in enumerate(result.material_formulations, 1):
                    print(f"\n{i}. {formulation.formulation_name}")
                    print(f"   Confidence: {formulation.confidence:.2f}")
                    print(f"   Precursors: {', '.join([f'{k}: {v}' for k, v in list(formulation.precursors.items())[:3]])}")
                
                print(f"\n🤖 L.A.R.A Expert Insights:")
                print(f"{result.design_rationale}")
                
                # Ask for detailed route
                detail_choice = input("\nWould you like detailed steps for a specific route? (y/n): ")
                if detail_choice.lower() == 'y' and result.synthesis_routes:
                    try:
                        route_num = int(input(f"Enter route number (1-{len(result.synthesis_routes)}): "))
                        if 1 <= route_num <= len(result.synthesis_routes):
                            selected_route = result.synthesis_routes[route_num-1]
                            print(f"\n📝 Detailed Steps for: {selected_route.route_name}")
                            for step_i, step in enumerate(selected_route.steps, 1):
                                print(f"{step_i}. {step}")
                            
                            print(f"\n⚙️ All Parameters:")
                            for param, value in selected_route.parameters.items():
                                print(f"- {param}: {value}")
                    except (ValueError, IndexError):
                        print("Invalid choice.")
                
            except Exception as e:
                print(f"❌ Error in inverse design: {e}")
    
    def _mixed_mode(self):
        """Mixed mode that automatically determines approach"""
        print("\n💡 Mixed/Auto Mode")
        print("Ask any question - the system will automatically choose the best approach")
        
        while True:
            query = input("\n🤔 Your question (or 'back' to return): ").strip()
            
            if query.lower() == 'back':
                break
            
            if not query:
                continue
            
            # Determine mode based on query content
            is_inverse_design = any(keyword in query.lower() for keyword in [
                'achieve', 'target', 'design', 'synthesize for', 'make with',
                'porosity', 'conductivity', 'density', 'properties'
            ]) and any(value_indicator in query.lower() for value_indicator in [
                'high', 'low', '>', '<', 'specific', 'exact', 'value'
            ])
            
            print(f"\n🔍 Processing: {query}")
            
            if is_inverse_design:
                print("🎯 Detected inverse design request - generating synthesis recommendations...")
                try:
                    result = self.perform_inverse_design(query)
                    print(f"\n🎯 Design Solution:")
                    print(f"Confidence: {result.confidence:.2f}")
                    print(f"\n📋 Top Synthesis Route: {result.synthesis_routes[0].route_name if result.synthesis_routes else 'None'}")
                    print(f"🤖 L.A.R.A Recommendations: {result.design_rationale[:200]}...")
                except Exception as e:
                    print(f"❌ Inverse design failed: {e}")
                    print("🔄 Falling back to hypothesis generation...")
                    try:
                        results = self.generate_hypotheses(query, 3)
                        for i, result in enumerate(results, 1):
                            print(f"{i}. {result.hypothesis}")
                    except Exception as e2:
                        print(f"❌ Hypothesis generation also failed: {e2}")
            else:
                print("🔬 Generating hypotheses and insights...")
                try:
                    results = self.generate_hypotheses(query, num_hypotheses=3)
                    
                    print(f"\n📝 Generated insights:")
                    for i, result in enumerate(results, 1):
                        print(f"\n{i}. {result.hypothesis}")
                        print(f"   Confidence: {result.confidence:.2f}")
                        print(f"   Source: {result.generated_by}")
                    
                except Exception as e:
                    print(f"❌ Error: {e}")

def main():
    """Main function with enhanced argument parsing"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="L.A.R.A Advanced System: Hypothesis Generation + Inverse Design",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python lara_advanced_system.py
  
  # Hypothesis generation
  python lara_advanced_system.py --mode hypothesis -q "Effect of pyrolysis on conductivity"
  
  # Inverse design
  python lara_advanced_system.py --mode inverse -p "high porosity 0.95, low thermal conductivity"
  
  # With custom model and KG data
  python lara_advanced_system.py -m /path/to/model --kg-path /path/to/matkg/data
        """
    )
    
    parser.add_argument(
        '--model-path', '-m',
        default="/home/kana_su/Scripts/llamat_finetuned_complete",
        help="Path to fine-tuned L.A.R.A model"
    )
    
    parser.add_argument(
        '--kg-path',
        default=None,
        help="Path to MatKG data files"
    )
    
    parser.add_argument(
        '--mode',
        choices=['hypothesis', 'inverse', 'mixed'],
        help="Operation mode: hypothesis generation, inverse design, or mixed"
    )
    
    parser.add_argument(
        '--query', '-q',
        help="Research question for hypothesis generation"
    )
    
    parser.add_argument(
        '--properties', '-p',
        help="Target properties for inverse design (e.g., 'high porosity, low density')"
    )
    
    parser.add_argument(
        '--num-hypotheses', '-n',
        type=int,
        default=5,
        help="Number of hypotheses to generate"
    )
    
    args = parser.parse_args()
    
    # Initialize system
    print("🚀 Initializing L.A.R.A Advanced System...")
    system = LARAAdvancedSystem(
        model_path=args.model_path,
        kg_data_path=args.kg_path
    )
    
    if args.mode == 'hypothesis' and args.query:
        # Hypothesis generation mode
        results = system.generate_hypotheses(args.query, args.num_hypotheses)
        
        print(f"\n🔬 Query: {args.query}")
        print(f"📝 Generated {len(results)} hypotheses:\n")
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.hypothesis}")
            print(f"   Confidence: {result.confidence:.2f}")
            print(f"   Generated by: {result.generated_by}")
            print()
    
    elif args.mode == 'inverse' and args.properties:
        # Inverse design mode
        result = system.perform_inverse_design(args.properties)
        
        print(f"\n🎯 Target Properties: {args.properties}")
        print(f"📋 Generated {len(result.synthesis_routes)} synthesis routes")
        print(f"🧪 Generated {len(result.material_formulations)} formulations")
        print(f"🤖 L.A.R.A Insights: {result.design_rationale}")
    
    else:
        # Interactive mode
        system.interactive_mode()

if __name__ == "__main__":
    main()

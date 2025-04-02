import * as THREE from 'three';

class ParticleSystem {
    constructor(count = 200, radius = 3.5, parent) {
        this.count = count;
        this.radius = radius;
        this.parent = parent;
        this.particles = [];
        this.group = new THREE.Group();
        this.isRecording = false;
        
        // Create a reusable geometry for all particles
        this.geometry = new THREE.BoxGeometry(0.05, 0.05, 0.05);
        
        // Define the central gap radius (matching the mesh size)
        this.centralGapRadius = 1.0; // This matches the mesh radius
        
        // Store original colors for each particle
        this.originalColors = [];
        
        this.init();
    }
    
    init() {
        // Create individual particles
        for (let i = 0; i < this.count; i++) {
            // Random orbit parameters with minimum distance from center
            // This ensures particles orbit around the central gap
            const minRadius = this.centralGapRadius * 1.2; // Slightly larger than the mesh
            const orbitRadius = minRadius + Math.random() * (this.radius - minRadius);
            
            const orbitSpeed = 0.2 + Math.random() * 0.8; // Range of speeds
            const orbitAngle = Math.random() * Math.PI * 2;
            
            // Random initial position on orbit
            const phi = Math.random() * Math.PI * 2;
            const theta = Math.random() * Math.PI;
            
            // Convert to cartesian coordinates
            const x = orbitRadius * Math.sin(theta) * Math.cos(phi);
            const y = orbitRadius * Math.sin(theta) * Math.sin(phi);
            const z = orbitRadius * Math.cos(theta);
            
            // Create a unique blue color for each particle
            const hue = 180 + Math.random() * 60; // Blue range in HSL
            const lightness = 40 + Math.random() * 40; // Varied lightness
            const color = new THREE.Color().setHSL(hue/360, 0.7, lightness/100);
            
            // Store original color
            this.originalColors.push({
                hue: hue,
                lightness: lightness
            });
            
            // Create material with blue color
            const material = new THREE.MeshBasicMaterial({ 
                color: color,
                transparent: true,
                opacity: 0.6
            });
            
            // Create mesh and set position
            const particle = new THREE.Mesh(this.geometry, material);
            particle.position.set(x, y, z);
            
            // Store additional data for animation
            particle.userData = {
                orbitRadius,
                orbitSpeed,
                orbitAngle,
                initialTheta: theta,
                initialPhi: phi,
                hue,
                lightness,
                pulseSpeed: 0.5 + Math.random() * 1.5,
                pulsePhase: Math.random() * Math.PI * 2
            };
            
            this.particles.push(particle);
            this.group.add(particle);
        }
        
        // Add the particle group to the parent
        if (this.parent) {
            this.parent.add(this.group);
        }
    }
    
    // Set recording state
    setRecording(isRecording) {
        this.isRecording = isRecording;
        
        // Apply immediate color change to all particles
        if (isRecording) {
            this.particles.forEach((particle, i) => {
                // Set to red with some variation
                const hue = 0; // Red hue
                const lightness = 40 + Math.random() * 30; // Varied lightness
                particle.material.color.setHSL(hue, 0.9, lightness/100);
            });
        } else {
            // Reset to original colors
            this.particles.forEach((particle, i) => {
                const data = this.originalColors[i];
                particle.material.color.setHSL(data.hue/360, 0.7, data.lightness/100);
            });
        }
    }
    
    update(time) {
        // Update each particle
        this.particles.forEach((particle, i) => {
            const data = particle.userData;
            
            // Calculate orbit movement
            // Use different rotation axes for variety
            const rotationX = time * 0.2 * data.orbitSpeed;
            const rotationY = time * 0.3 * data.orbitSpeed + data.orbitAngle;
            const rotationZ = time * 0.1 * data.orbitSpeed;
            
            // Create rotation matrices for complex orbit paths
            const rotMatrix = new THREE.Matrix4().makeRotationX(rotationX)
                .multiply(new THREE.Matrix4().makeRotationY(rotationY))
                .multiply(new THREE.Matrix4().makeRotationZ(rotationZ));
            
            // Convert spherical to cartesian with rotation
            const position = new THREE.Vector3(
                data.orbitRadius * Math.sin(data.initialTheta) * Math.cos(data.initialPhi),
                data.orbitRadius * Math.sin(data.initialTheta) * Math.sin(data.initialPhi),
                data.orbitRadius * Math.cos(data.initialTheta)
            );
            
            // Apply rotation
            position.applyMatrix4(rotMatrix);
            
            // Set new position
            particle.position.copy(position);
            
            // Pulse color and scale
            if (!this.isRecording) {
                // Normal blue mode
                const pulse = Math.sin(time * data.pulseSpeed + data.pulsePhase) * 0.5 + 0.5;
                const hue = data.hue + pulse * 10; // Subtle hue shift
                const lightness = data.lightness + pulse * 15; // Brightness pulse
                
                // Update color
                particle.material.color.setHSL(hue/360, 0.8, lightness/100);
            } else {
                // Recording red mode
                const pulse = Math.sin(time * data.pulseSpeed * 2 + data.pulsePhase) * 0.5 + 0.5;
                const hue = 0; // Red base
                const lightness = 40 + pulse * 40; // More dramatic brightness pulse
                
                // Update color
                particle.material.color.setHSL(hue, 0.9, lightness/100);
            }
            
            // Scale particles slightly based on pulse
            const pulse = Math.sin(time * data.pulseSpeed + data.pulsePhase) * 0.5 + 0.5;
            const scale = this.isRecording ? 
                (0.8 + pulse * 0.8) : // Larger pulse when recording
                (0.8 + pulse * 0.4);  // Normal pulse otherwise
            particle.scale.set(scale, scale, scale);
        });
    }
    
    // Method to respond to audio input
    respondToAudio(frequency) {
        const normalizedFreq = Math.min(frequency / 200, 1);
        
        this.particles.forEach((particle, i) => {
            // Increase particle speed based on audio
            const boost = 1 + normalizedFreq * 3;
            particle.userData.currentSpeed = particle.userData.orbitSpeed * boost;
            
            // Increase brightness with frequency
            const material = particle.material;
            const data = particle.userData;
            
            if (!this.isRecording) {
                // Normal blue mode
                material.color.setHSL(
                    data.hue/360, 
                    0.8, 
                    Math.min(100, data.lightness + normalizedFreq * 40)/100
                );
            } else {
                // Recording red mode
                material.color.setHSL(
                    0, // Red hue
                    0.9,
                    Math.min(100, 40 + normalizedFreq * 60)/100
                );
            }
            
            // Audio response can also affect the particle distances from center
            // Pull particles slightly toward or away from the center based on frequency
            if (normalizedFreq > 0.5) {
                // Create a pulse effect that preserves the central gap
                const pulseEffect = (normalizedFreq - 0.5) * 0.2; // Up to 10% change
                const currentRadius = particle.userData.orbitRadius;
                const pulsedRadius = currentRadius * (1 + pulseEffect * Math.sin(data.pulsePhase));
                
                // Ensure we never go below the central gap radius
                const safeRadius = Math.max(this.centralGapRadius * 1.2, pulsedRadius);
                
                // Apply this temporary radius change
                const direction = particle.position.clone().normalize();
                particle.position.copy(direction.multiplyScalar(safeRadius));
            }
        });
    }
}

export default ParticleSystem;
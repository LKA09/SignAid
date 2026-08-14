import React, { useEffect, useRef } from 'react'
import * as THREE from './vendor/three.module.js'

const BODY_SEGMENTS = [
  [10, 11, 'pants', 0.085], [11, 12, 'pants', 0.072],
  [13, 14, 'pants', 0.085], [14, 15, 'pants', 0.072],
  [2, 3, 'shirt', 0.078], [5, 6, 'shirt', 0.078],
  [3, 4, 'skin', 0.062], [6, 7, 'skin', 0.062],
]

const HAND_SEGMENTS = [
  [17, 18], [18, 19], [19, 20], [20, 21],
  [17, 22], [22, 23], [23, 24], [24, 25],
  [17, 26], [26, 27], [27, 28], [28, 29],
  [17, 30], [30, 31], [31, 32], [32, 33],
  [17, 34], [34, 35], [35, 36], [36, 37],
  [38, 39], [39, 40], [40, 41], [41, 42],
  [38, 43], [43, 44], [44, 45], [45, 46],
  [38, 47], [47, 48], [48, 49], [49, 50],
  [38, 51], [51, 52], [52, 53], [53, 54],
  [38, 55], [55, 56], [56, 57], [57, 58],
]

const Y_AXIS = new THREE.Vector3(0, 1, 0)

function point(frame, index) {
  const value = frame[index]
  return new THREE.Vector3(value[0], value[1], value[2])
}

function placeBetween(mesh, a, b) {
  const direction = new THREE.Vector3().subVectors(b, a)
  const length = Math.max(direction.length(), 0.001)
  mesh.position.copy(a).add(b).multiplyScalar(0.5)
  mesh.scale.set(1, length, 1)
  mesh.quaternion.setFromUnitVectors(Y_AXIS, direction.normalize())
}

export default function Avatar3D({ motion, fps = 20 }) {
  const hostRef = useRef(null)

  useEffect(() => {
    if (!hostRef.current || !motion?.length) return undefined

    const host = hostRef.current
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x071522)
    scene.fog = new THREE.Fog(0x071522, 3.6, 6)

    const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 20)
    camera.position.set(0, 1.05, 3.55)
    camera.lookAt(0, 1.03, 0)

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFSoftShadowMap
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.12
    host.appendChild(renderer.domElement)

    scene.add(new THREE.HemisphereLight(0xbdefff, 0x142038, 2.2))
    const keyLight = new THREE.DirectionalLight(0xffffff, 3.2)
    keyLight.position.set(-2.5, 4, 3.5)
    keyLight.castShadow = true
    scene.add(keyLight)
    const rimLight = new THREE.DirectionalLight(0x2dd4bf, 2.4)
    rimLight.position.set(3, 2.2, -2)
    scene.add(rimLight)

    const floor = new THREE.Mesh(
      new THREE.CircleGeometry(1.25, 64),
      new THREE.MeshStandardMaterial({ color: 0x091a2b, roughness: 0.92, metalness: 0.05 }),
    )
    floor.rotation.x = -Math.PI / 2
    floor.position.y = -0.045
    floor.receiveShadow = true
    scene.add(floor)
    const grid = new THREE.GridHelper(2.35, 12, 0x27777a, 0x183c4f)
    grid.position.y = -0.038
    grid.material.transparent = true
    grid.material.opacity = 0.32
    scene.add(grid)

    const avatar = new THREE.Group()
    scene.add(avatar)

    const materials = {
      skin: new THREE.MeshStandardMaterial({ color: 0xd99a79, roughness: 0.72 }),
      skinLight: new THREE.MeshStandardMaterial({ color: 0xefb797, roughness: 0.68 }),
      shirt: new THREE.MeshStandardMaterial({ color: 0x258c92, roughness: 0.6 }),
      pants: new THREE.MeshStandardMaterial({ color: 0x203b5a, roughness: 0.75 }),
      hair: new THREE.MeshStandardMaterial({ color: 0x142235, roughness: 0.85 }),
      dark: new THREE.MeshStandardMaterial({ color: 0x07111f, roughness: 0.7 }),
      white: new THREE.MeshStandardMaterial({ color: 0xf2ffff, roughness: 0.55 }),
    }

    function mesh(geometry, material) {
      const object = new THREE.Mesh(geometry, material)
      object.castShadow = true
      object.receiveShadow = true
      avatar.add(object)
      return object
    }

    function segment(radius, material) {
      return mesh(new THREE.CylinderGeometry(radius, radius * 1.02, 1, 14), material)
    }

    const bodySegments = BODY_SEGMENTS.map(([a, b, material, radius]) => ({
      a, b, object: segment(radius, materials[material]),
    }))
    const fingerSegments = HAND_SEGMENTS.map(([a, b]) => ({
      a, b, object: segment(0.014, materials.skinLight),
    }))

    const torso = mesh(new THREE.CapsuleGeometry(0.245, 0.38, 8, 20), materials.shirt)
    torso.scale.z = 0.56
    const hips = mesh(new THREE.CapsuleGeometry(0.18, 0.16, 6, 18), materials.pants)
    hips.scale.z = 0.62
    const neck = segment(0.07, materials.skin)

    const head = new THREE.Group()
    avatar.add(head)
    const face = new THREE.Mesh(new THREE.SphereGeometry(0.14, 32, 24), materials.skinLight)
    face.scale.set(0.9, 1.08, 0.88)
    face.castShadow = true
    head.add(face)
    const hair = new THREE.Mesh(
      new THREE.SphereGeometry(0.145, 32, 16, 0, Math.PI * 2, 0, Math.PI * 0.54),
      materials.hair,
    )
    hair.scale.set(0.94, 1.06, 0.93)
    hair.position.y = 0.026
    head.add(hair)
    for (const x of [-0.047, 0.047]) {
      const eye = new THREE.Mesh(new THREE.SphereGeometry(0.010, 12, 8), materials.dark)
      eye.position.set(x, 0.025, 0.119)
      head.add(eye)
    }
    const nose = new THREE.Mesh(new THREE.SphereGeometry(0.011, 12, 8), materials.skin)
    nose.position.set(0, -0.012, 0.134)
    nose.scale.set(0.72, 1.2, 0.9)
    head.add(nose)
    const mouth = new THREE.Mesh(new THREE.BoxGeometry(0.055, 0.008, 0.008), new THREE.MeshStandardMaterial({ color: 0x984d56 }))
    mouth.position.set(0, -0.067, 0.124)
    head.add(mouth)

    const palms = [
      mesh(new THREE.SphereGeometry(0.058, 18, 12), materials.skinLight),
      mesh(new THREE.SphereGeometry(0.058, 18, 12), materials.skinLight),
    ]
    palms.forEach(palm => palm.scale.set(0.95, 1.2, 0.42))

    const jointSpheres = [3, 4, 6, 7, 11, 12, 14, 15].map((index) => ({
      index,
      object: mesh(
        new THREE.SphereGeometry(index === 4 || index === 7 ? 0.055 : 0.07, 16, 12),
        index === 11 || index === 12 || index === 14 || index === 15 ? materials.pants : materials.skin,
      ),
    }))
    const shoes = [12, 15].map(index => ({
      index,
      object: mesh(new THREE.CapsuleGeometry(0.06, 0.12, 5, 14), materials.dark),
    }))
    shoes.forEach(({ object }) => {
      object.rotation.x = Math.PI / 2
      object.scale.set(1.05, 1, 0.75)
    })

    const resize = () => {
      const width = Math.max(host.clientWidth, 1)
      const height = Math.max(host.clientHeight, 1)
      renderer.setSize(width, height, false)
      camera.aspect = width / height
      camera.updateProjectionMatrix()
    }
    const observer = new ResizeObserver(resize)
    observer.observe(host)
    resize()

    let dragging = false
    let previousX = 0
    let rotationY = 0
    const onPointerDown = event => {
      dragging = true
      previousX = event.clientX
      renderer.domElement.setPointerCapture(event.pointerId)
    }
    const onPointerMove = event => {
      if (!dragging) return
      rotationY += (event.clientX - previousX) * 0.012
      previousX = event.clientX
    }
    const onPointerUp = () => { dragging = false }
    renderer.domElement.addEventListener('pointerdown', onPointerDown)
    renderer.domElement.addEventListener('pointermove', onPointerMove)
    renderer.domElement.addEventListener('pointerup', onPointerUp)
    renderer.domElement.addEventListener('pointercancel', onPointerUp)

    const start = performance.now()
    let animationId = 0
    const renderFrame = now => {
      const elapsed = Math.max(0, now - start)
      const candidate = Math.floor((elapsed / 1000) * fps) % motion.length
      const frameIndex = Number.isFinite(candidate) && candidate >= 0 ? candidate : 0
      const frame = motion[frameIndex] || motion[0]
      bodySegments.forEach(({ a, b, object }) => placeBetween(object, point(frame, a), point(frame, b)))
      fingerSegments.forEach(({ a, b, object }) => placeBetween(object, point(frame, a), point(frame, b)))

      const shoulderCenter = point(frame, 2).add(point(frame, 5)).multiplyScalar(0.5)
      const hipCenter = point(frame, 10).add(point(frame, 13)).multiplyScalar(0.5)
      torso.position.copy(shoulderCenter).lerp(hipCenter, 0.5)
      torso.position.y += 0.015
      hips.position.copy(hipCenter)
      hips.position.y += 0.025
      placeBetween(neck, point(frame, 1), point(frame, 0).add(new THREE.Vector3(0, -0.09, 0)))
      head.position.copy(point(frame, 0))

      palms[0].position.copy(point(frame, 17)).lerp(point(frame, 26), 0.42)
      palms[1].position.copy(point(frame, 38)).lerp(point(frame, 47), 0.42)
      jointSpheres.forEach(({ index, object }) => object.position.copy(point(frame, index)))
      shoes.forEach(({ index, object }) => {
        object.position.copy(point(frame, index))
        object.position.z += 0.045
        object.position.y += 0.015
      })

      avatar.rotation.y = rotationY + Math.sin((now - start) * 0.00045) * 0.045
      renderer.render(scene, camera)
      animationId = requestAnimationFrame(renderFrame)
    }
    animationId = requestAnimationFrame(renderFrame)

    return () => {
      cancelAnimationFrame(animationId)
      observer.disconnect()
      renderer.domElement.removeEventListener('pointerdown', onPointerDown)
      renderer.domElement.removeEventListener('pointermove', onPointerMove)
      renderer.domElement.removeEventListener('pointerup', onPointerUp)
      renderer.domElement.removeEventListener('pointercancel', onPointerUp)
      scene.traverse(object => object.geometry?.dispose())
      Object.values(materials).forEach(material => material.dispose())
      renderer.dispose()
      renderer.domElement.remove()
    }
  }, [motion, fps])

  return <div ref={hostRef} className="avatar-3d" aria-label="회전 가능한 3D 수어 아바타">
    <span className="drag-hint">드래그해서 회전</span>
  </div>
}

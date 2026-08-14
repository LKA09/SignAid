import React, { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from './vendor/three.module.js'
import { GLTFLoader } from './vendor/loaders/GLTFLoader.js'

const API = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '')
const MODEL_URL = `${API}/avatar-assets/stable/AvatarSample_C.vrm`

const BONE_TARGETS = {
  spine: [9, 8], chest: [8, 16], upperChest: [16, 1], neck: [1, 0],
  // Source coordinates are camera-relative, so their visual left/right is
  // opposite the avatar's anatomical left/right.
  leftUpperLeg: [13, 14], leftLowerLeg: [14, 15],
  rightUpperLeg: [10, 11], rightLowerLeg: [11, 12],
  leftUpperArm: [5, 6], leftLowerArm: [6, 7], leftHand: [38, 47],
  rightUpperArm: [2, 3], rightLowerArm: [3, 4], rightHand: [17, 26],
  leftThumbProximal: [39, 40], leftThumbIntermediate: [40, 41], leftThumbDistal: [41, 42],
  leftIndexProximal: [43, 44], leftIndexIntermediate: [44, 45], leftIndexDistal: [45, 46],
  leftMiddleProximal: [47, 48], leftMiddleIntermediate: [48, 49], leftMiddleDistal: [49, 50],
  leftRingProximal: [51, 52], leftRingIntermediate: [52, 53], leftRingDistal: [53, 54],
  leftLittleProximal: [55, 56], leftLittleIntermediate: [56, 57], leftLittleDistal: [57, 58],
  rightThumbProximal: [18, 19], rightThumbIntermediate: [19, 20], rightThumbDistal: [20, 21],
  rightIndexProximal: [22, 23], rightIndexIntermediate: [23, 24], rightIndexDistal: [24, 25],
  rightMiddleProximal: [26, 27], rightMiddleIntermediate: [27, 28], rightMiddleDistal: [28, 29],
  rightRingProximal: [30, 31], rightRingIntermediate: [31, 32], rightRingDistal: [32, 33],
  rightLittleProximal: [34, 35], rightLittleIntermediate: [35, 36], rightLittleDistal: [36, 37],
}

const CHILD_BONES = {
  spine: 'chest', chest: 'upperChest', upperChest: 'neck', neck: 'head',
  leftUpperLeg: 'leftLowerLeg', leftLowerLeg: 'leftFoot',
  rightUpperLeg: 'rightLowerLeg', rightLowerLeg: 'rightFoot',
  leftUpperArm: 'leftLowerArm', leftLowerArm: 'leftHand', leftHand: 'leftMiddleProximal',
  rightUpperArm: 'rightLowerArm', rightLowerArm: 'rightHand', rightHand: 'rightMiddleProximal',
  leftThumbProximal: 'leftThumbIntermediate', leftThumbIntermediate: 'leftThumbDistal',
  leftIndexProximal: 'leftIndexIntermediate', leftIndexIntermediate: 'leftIndexDistal',
  leftMiddleProximal: 'leftMiddleIntermediate', leftMiddleIntermediate: 'leftMiddleDistal',
  leftRingProximal: 'leftRingIntermediate', leftRingIntermediate: 'leftRingDistal',
  leftLittleProximal: 'leftLittleIntermediate', leftLittleIntermediate: 'leftLittleDistal',
  rightThumbProximal: 'rightThumbIntermediate', rightThumbIntermediate: 'rightThumbDistal',
  rightIndexProximal: 'rightIndexIntermediate', rightIndexIntermediate: 'rightIndexDistal',
  rightMiddleProximal: 'rightMiddleIntermediate', rightMiddleIntermediate: 'rightMiddleDistal',
  rightRingProximal: 'rightRingIntermediate', rightRingIntermediate: 'rightRingDistal',
  rightLittleProximal: 'rightLittleIntermediate', rightLittleIntermediate: 'rightLittleDistal',
}

function smootherStep(value) {
  const t = THREE.MathUtils.clamp(value, 0, 1)
  return t * t * t * (t * (t * 6 - 15) + 10)
}

function appendLoopBridge(frames) {
  if (!frames?.length || frames.length < 3) return frames || []
  const bridgeLength = Math.min(10, Math.max(4, Math.round(frames.length * 0.08)))
  const first = frames[0]
  const last = frames[frames.length - 1]
  const interpolate = (from, to, mix) => Array.isArray(from)
    ? from.map((value, index) => interpolate(value, to?.[index] ?? value, mix))
    : THREE.MathUtils.lerp(Number(from) || 0, Number(to) || 0, mix)
  const bridge = Array.from({ length: bridgeLength }, (_, bridgeIndex) => (
    interpolate(last, first, smootherStep((bridgeIndex + 1) / (bridgeLength + 1)))
  ))
  return [...frames, ...bridge]
}

function prepareMotion(frames) {
  if (!frames?.length) return []
  const smoothed = frames.map((frame, frameIndex) => frame.map((point, jointIndex) => {
    const weights = jointIndex >= 17 ? [1, 2, 3, 4, 3, 2, 1] : [1, 2, 3, 2, 1]
    const radius = Math.floor(weights.length / 2)
    const sum = [0, 0, 0]
    let totalWeight = 0
    weights.forEach((weight, weightIndex) => {
      const sampleIndex = THREE.MathUtils.clamp(frameIndex + weightIndex - radius, 0, frames.length - 1)
      const sample = frames[sampleIndex]?.[jointIndex] || point
      for (let axis = 0; axis < 3; axis += 1) {
        sum[axis] += (Number(sample?.[axis]) || 0) * weight
      }
      totalWeight += weight
    })
    return sum.map(value => value / totalWeight)
  }))

  return appendLoopBridge(smoothed)
}

function motionPoint(frames, frameIndex, nextFrameIndex, index, mix, target) {
  const current = frames[frameIndex]?.[index] || [0, 0, 0]
  const next = frames[nextFrameIndex]?.[index] || current
  return target.set(
    THREE.MathUtils.lerp(current[0], next[0], mix),
    THREE.MathUtils.lerp(current[1], next[1], mix),
    THREE.MathUtils.lerp(current[2], next[2], mix),
  )
}

function interpolatedValues(frames, frameIndex, nextFrameIndex, mix, fallback) {
  const current = frames[frameIndex] || fallback
  const next = frames[nextFrameIndex] || current
  return current.map((value, index) => THREE.MathUtils.lerp(value, next[index] ?? value, mix))
}

function retargetProfile(name, handRetargeting) {
  if (/Thumb|Index|Middle|Ring|Little/.test(name)) {
    const maxDeviation = name.endsWith('Distal') ? 0.34 : name.endsWith('Intermediate') ? 0.44 : 0.58
    return handRetargeting
      ? { strength: 0.46, response: 6.2, maxSpeed: 1.75, maxDeviation }
      : { strength: 0, response: 12, maxSpeed: 3.2, maxDeviation: 0 }
  }
  if (name.endsWith('Hand')) {
    return handRetargeting
      ? { strength: 0.42, response: 7, maxSpeed: 2.1, maxDeviation: 0.5 }
      : { strength: 0, response: 12, maxSpeed: 3.2, maxDeviation: 0 }
  }
  if (name.includes('Leg')) return { strength: 0.08, response: 4, maxSpeed: 0.8, maxDeviation: 0.25 }
  if (/spine|chest|upperChest|neck/.test(name)) {
    return { strength: 0.1, response: 4.5, maxSpeed: 0.8, maxDeviation: 0.3 }
  }
  if (name.endsWith('UpperArm')) return { strength: 0.78, response: 7, maxSpeed: 2.35, maxDeviation: 2.2 }
  if (name.endsWith('LowerArm')) return { strength: 0.9, response: 8, maxSpeed: 3, maxDeviation: 2.35 }
  return { strength: 0.5, response: 7, maxSpeed: 2, maxDeviation: 1.2 }
}

export default function VrmAvatar3D({
  motion, palmNormals = [], facialExpressions = [], headRotations = [],
  fps = 20, motionSource = '', playbackRate = 0.72, onError,
}) {
  const hostRef = useRef(null)
  const [status, setStatus] = useState('loading')
  const preparedMotion = useMemo(() => prepareMotion(motion), [motion])
  const preparedPalmNormals = useMemo(() => appendLoopBridge(palmNormals), [palmNormals])
  const preparedFacialExpressions = useMemo(() => appendLoopBridge(facialExpressions), [facialExpressions])
  const preparedHeadRotations = useMemo(() => appendLoopBridge(headRotations), [headRotations])
  const handRetargeting = motionSource.includes('aihub_')

  useEffect(() => {
    if (!hostRef.current || !preparedMotion.length) return undefined
    let disposed = false
    const host = hostRef.current
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x281817)
    scene.fog = new THREE.Fog(0x281817, 4.0, 7.0)

    const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 20)
    camera.position.set(0, 1.23, 2.42)
    camera.lookAt(0, 1.2, 0)

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    renderer.shadowMap.enabled = true
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.2
    host.appendChild(renderer.domElement)

    scene.add(new THREE.HemisphereLight(0xfff0d8, 0x4b211e, 2.5))
    const key = new THREE.DirectionalLight(0xffe8cf, 3.4)
    key.position.set(-2.2, 4.2, 3.8)
    key.castShadow = true
    scene.add(key)
    const fill = new THREE.DirectionalLight(0xc95b51, 2.2)
    fill.position.set(3.2, 2.4, -2.8)
    scene.add(fill)

    const floor = new THREE.Mesh(
      new THREE.CircleGeometry(1.3, 64),
      new THREE.MeshStandardMaterial({ color: 0x38211f, roughness: 0.92 }),
    )
    floor.rotation.x = -Math.PI / 2
    floor.position.y = -0.025
    floor.receiveShadow = true
    scene.add(floor)
    const grid = new THREE.GridHelper(2.45, 12, 0xb85b50, 0x63342f)
    grid.position.y = -0.018
    grid.material.transparent = true
    grid.material.opacity = 0.28
    scene.add(grid)

    const avatarRoot = new THREE.Group()
    scene.add(avatarRoot)
    let rig = []
    let headController = null
    let expressionBindings = {}
    let rotationY = 0

    const loader = new GLTFLoader()
    loader.load(MODEL_URL, async gltf => {
      if (disposed) return
      const model = gltf.scene
      model.rotation.y = Math.PI
      model.traverse(object => {
        if (object.isMesh) {
          object.castShadow = true
          object.receiveShadow = true
          object.frustumCulled = false
        }
      })
      avatarRoot.add(model)
      model.updateWorldMatrix(true, true)

      let bounds = new THREE.Box3().setFromObject(model)
      const height = Math.max(bounds.max.y - bounds.min.y, 0.001)
      model.scale.multiplyScalar(1.62 / height)
      model.updateWorldMatrix(true, true)
      bounds = new THREE.Box3().setFromObject(model)
      model.position.y -= bounds.min.y
      model.updateWorldMatrix(true, true)

      const extension = gltf.parser.json.extensions?.VRM
      const humanBones = extension?.humanoid?.humanBones || []
      const boneEntries = await Promise.all(humanBones.map(async entry => [
        entry.bone,
        await gltf.parser.getDependency('node', entry.node),
      ]))
      const bones = Object.fromEntries(boneEntries)

      headController = bones.head ? {
        bone: bones.head,
        restLocalQuaternion: bones.head.quaternion.clone(),
      } : null

      const blendShapeGroups = extension?.blendShapeMaster?.blendShapeGroups || []
      expressionBindings = {}
      for (const group of blendShapeGroups) {
        const bindings = []
        for (const bind of group.binds || []) {
          model.traverse(object => {
            const association = gltf.parser.associations?.get(object)
            if (association?.meshes === bind.mesh && object.morphTargetInfluences?.[bind.index] != null) {
              bindings.push({ object, index: bind.index, weight: (bind.weight ?? 100) / 100 })
            }
          })
        }
        const names = [group.presetName, group.name].filter(Boolean).map(value => value.toLowerCase())
        for (const name of names) expressionBindings[name] = bindings
      }

      model.updateWorldMatrix(true, true)
      rig = Object.entries(BONE_TARGETS).flatMap(([name, target]) => {
        const bone = bones[name]
        const child = bones[CHILD_BONES[name]]
        if (!bone || !child) return []
        const origin = bone.getWorldPosition(new THREE.Vector3())
        const childPosition = child.getWorldPosition(new THREE.Vector3())
        const restDirection = childPosition.sub(origin).normalize()
        const restWorldQuaternion = bone.getWorldQuaternion(new THREE.Quaternion())
        const restLocalQuaternion = bone.quaternion.clone()
        const isHand = name === 'leftHand' || name === 'rightHand'
        if (isHand && !preparedPalmNormals.length) {
          const palmTurn = name === 'leftHand' ? Math.PI / 2 : -Math.PI / 2
          restLocalQuaternion.multiply(
            new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), palmTurn),
          )
        }
        let palmIndex = null
        let restPalmBasisInverse = null
        if (isHand) {
          const side = name.startsWith('left') ? 'left' : 'right'
          const indexBone = bones[`${side}IndexProximal`]
          const littleBone = bones[`${side}LittleProximal`]
          if (indexBone && littleBone) {
            const indexPosition = indexBone.getWorldPosition(new THREE.Vector3())
            const littlePosition = littleBone.getWorldPosition(new THREE.Vector3())
            const across = littlePosition.sub(indexPosition)
            across.addScaledVector(restDirection, -across.dot(restDirection)).normalize()
            const normal = new THREE.Vector3().crossVectors(across, restDirection).normalize()
            const matrix = new THREE.Matrix4().makeBasis(across, restDirection, normal)
            restPalmBasisInverse = new THREE.Quaternion().setFromRotationMatrix(matrix).invert()
            palmIndex = side === 'right' ? 0 : 1
          }
        }
        const profile = retargetProfile(name, handRetargeting)
        return [{
          bone, target, restDirection, restWorldQuaternion, restLocalQuaternion,
          palmIndex, restPalmBasisInverse, ...profile,
        }]
      })
      setStatus(rig.length >= 30 ? 'ready' : 'limited')
    }, undefined, error => {
      console.error('VRM avatar load failed', error)
      setStatus('error')
      onError?.(error)
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
    let previousTime = start
    let animationId = 0
    const parentWorld = new THREE.Quaternion()
    const delta = new THREE.Quaternion()
    const desiredWorld = new THREE.Quaternion()
    const desiredLocal = new THREE.Quaternion()
    const targetLocal = new THREE.Quaternion()
    const fromPoint = new THREE.Vector3()
    const toPoint = new THREE.Vector3()
    const targetDirection = new THREE.Vector3()
    const palmNormal = new THREE.Vector3()
    const targetAcross = new THREE.Vector3()
    const palmBasisMatrix = new THREE.Matrix4()
    const palmBasisQuaternion = new THREE.Quaternion()
    const headCurrent = new THREE.Quaternion()
    const headNext = new THREE.Quaternion()
    const headRelative = new THREE.Quaternion()
    const headTarget = new THREE.Quaternion()
    const setExpression = (name, value) => {
      for (const binding of expressionBindings[name] || []) {
        binding.object.morphTargetInfluences[binding.index] = THREE.MathUtils.clamp(value * binding.weight, 0, 1)
      }
    }
    const renderFrame = now => {
      const elapsed = Math.max(0, now - start)
      const deltaTime = THREE.MathUtils.clamp((now - previousTime) / 1000, 1 / 120, 1 / 15)
      previousTime = now
      const frameTime = (elapsed / 1000) * fps * playbackRate
      const wholeFrame = Math.floor(frameTime)
      const frameIndex = wholeFrame % preparedMotion.length
      const nextFrameIndex = (frameIndex + 1) % preparedMotion.length
      const frameMix = smootherStep(frameTime - wholeFrame)

      for (const item of rig) {
        targetLocal.copy(item.restLocalQuaternion)
        if (item.strength > 0) {
          const [from, to] = item.target
          motionPoint(preparedMotion, frameIndex, nextFrameIndex, from, frameMix, fromPoint)
          motionPoint(preparedMotion, frameIndex, nextFrameIndex, to, frameMix, toPoint)
          targetDirection.copy(toPoint).sub(fromPoint)
          if (targetDirection.lengthSq() < 1e-8) continue
          targetDirection.normalize()
          if (item.palmIndex != null && item.restPalmBasisInverse && preparedPalmNormals.length) {
            motionPoint(preparedPalmNormals, frameIndex, nextFrameIndex, item.palmIndex, frameMix, palmNormal)
            palmNormal.addScaledVector(targetDirection, -palmNormal.dot(targetDirection))
            if (palmNormal.lengthSq() > 1e-8) {
              palmNormal.normalize()
              targetAcross.crossVectors(targetDirection, palmNormal).normalize()
              palmNormal.crossVectors(targetAcross, targetDirection).normalize()
              palmBasisMatrix.makeBasis(targetAcross, targetDirection, palmNormal)
              palmBasisQuaternion.setFromRotationMatrix(palmBasisMatrix)
              delta.copy(palmBasisQuaternion).multiply(item.restPalmBasisInverse)
            } else {
              delta.setFromUnitVectors(item.restDirection, targetDirection)
            }
          } else {
            delta.setFromUnitVectors(item.restDirection, targetDirection)
          }
          desiredWorld.copy(delta).multiply(item.restWorldQuaternion)
          item.bone.parent.getWorldQuaternion(parentWorld).invert()
          desiredLocal.copy(parentWorld).multiply(desiredWorld)
          targetLocal.slerp(desiredLocal, item.strength)
          const deviation = item.restLocalQuaternion.angleTo(targetLocal)
          if (deviation > item.maxDeviation && deviation > 1e-5) {
            targetLocal.copy(item.restLocalQuaternion).slerp(targetLocal, item.maxDeviation / deviation)
          }
        }

        const angle = item.bone.quaternion.angleTo(targetLocal)
        const responsiveStep = 1 - Math.exp(-item.response * deltaTime)
        const speedStep = angle > 1e-5 ? (item.maxSpeed * deltaTime) / angle : 1
        item.bone.quaternion.slerp(targetLocal, Math.min(1, responsiveStep, speedStep))
        item.bone.updateWorldMatrix(false, true)
      }

      if (headController && preparedHeadRotations.length) {
        const current = preparedHeadRotations[frameIndex] || [0, 0, 0, 1]
        const next = preparedHeadRotations[nextFrameIndex] || current
        headCurrent.fromArray(current)
        headNext.fromArray(next)
        if (headCurrent.dot(headNext) < 0) headNext.set(-headNext.x, -headNext.y, -headNext.z, -headNext.w)
        headRelative.copy(headCurrent).slerp(headNext, frameMix)
        const headAngle = new THREE.Quaternion().angleTo(headRelative)
        if (headAngle > 0.55) {
          headRelative.identity().slerp(headCurrent.slerp(headNext, frameMix), 0.55 / headAngle)
        }
        headTarget.copy(headController.restLocalQuaternion).multiply(headRelative)
        headController.bone.quaternion.slerp(headTarget, 1 - Math.exp(-6 * deltaTime))
        headController.bone.updateWorldMatrix(false, true)
      }

      if (preparedFacialExpressions.length) {
        const expression = interpolatedValues(
          preparedFacialExpressions, frameIndex, nextFrameIndex, frameMix, [0, 0, 0, 0],
        )
        setExpression('a', expression[0] * 0.72)
        setExpression('blink_l', expression[1])
        setExpression('blink_r', expression[2])
        setExpression('surprised', expression[3] * 0.42)
      }

      avatarRoot.rotation.y = rotationY + Math.sin(elapsed * 0.0004) * 0.035
      avatarRoot.position.y = Math.sin(elapsed * 0.0018) * 0.004
      renderer.render(scene, camera)
      animationId = requestAnimationFrame(renderFrame)
    }
    animationId = requestAnimationFrame(renderFrame)

    return () => {
      disposed = true
      cancelAnimationFrame(animationId)
      observer.disconnect()
      renderer.domElement.removeEventListener('pointerdown', onPointerDown)
      renderer.domElement.removeEventListener('pointermove', onPointerMove)
      renderer.domElement.removeEventListener('pointerup', onPointerUp)
      renderer.domElement.removeEventListener('pointercancel', onPointerUp)
      scene.traverse(object => {
        object.geometry?.dispose()
        if (Array.isArray(object.material)) object.material.forEach(material => material.dispose())
        else object.material?.dispose()
      })
      renderer.dispose()
      renderer.domElement.remove()
    }
  }, [
    preparedMotion, preparedPalmNormals, preparedFacialExpressions, preparedHeadRotations,
    fps, handRetargeting, playbackRate, onError,
  ])

  return <div ref={hostRef} className="avatar-3d" aria-label="회전 가능한 VRM 3D 수어 아바타">
    <span className="drag-hint">드래그해서 회전</span>
    {status === 'loading' && <span className="model-status">자연스러운 3D 모델 불러오는 중…</span>}
    {status === 'limited' && <span className="model-status warning">일부 관절만 연결됨</span>}
    {status === 'error' && <span className="model-status warning">3D 모델을 불러오지 못했습니다.</span>}
  </div>
}
